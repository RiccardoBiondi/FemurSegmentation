#!/usr/bin/env python
# -*- coding: utf-8 -*-



import itk
import numpy as np

from FemurSegmentation.utils import cast_image
from FemurSegmentation.utils import array2image
from FemurSegmentation.utils import image2array
from FemurSegmentation.filters import median_filter
from FemurSegmentation.filters import connected_components
from FemurSegmentation.filters import binary_threshold
from FemurSegmentation.filters import opening
from FemurSegmentation.filters import execute_pipeline


__author__ = ['Riccardo Biondi']
__email__ = ['riccardo.biondi7@unibo.it']



## Rigurada bene il metodo e uniforma i nomi!! il camel case è necessario


class LegImages :

    def __init__ (self, image, mask = None) :

        arr, info = image2array(image)
        self.image = image
        self.mask = mask
        self.info = info
        self.shape = arr.shape # image shape to reconstruction

        pixelType, dim = itk.template(image)[1]

        self.ImageType = itk.Image[pixelType, dim]
        LabelType = itk.StatisticsLabelObject[itk.UL, dim]
        self.LabelMap = itk.LabelMap[LabelType]



    def _initRegion(self, indexes) :

        RegionType = itk.ImageRegion[3]
        region = RegionType()
        _ = region.SetIndex(indexes[0])
        _ = region.SetUpperIndex(indexes[1])

        return region

    def _initStartEndIndexes(self, start, end) :

        st = itk.Index[3]()
        st[0] = int(start[0])
        st[1] = int(start[1])
        st[2] = int(start[2])

        ed = itk.Index[3]()
        ed[0] = int(end[0])
        ed[1] = int(end[1])
        ed[2] = int(end[2])

        return st, ed

    def getBoundingBox(self, img) :

        imageType = itk.Image[itk.UC, 3]
        shape = itk.LabelImageToShapeLabelMapFilter[imageType,
                                                    self.LabelMap].New()
        _ = shape.SetInput(img)
        _ = shape.SetComputePerimeter(False)
        _ = shape.Update()

        labelMap = shape.GetOutput()
        return labelMap.GetNthLabelObject(0).GetBoundingBox()

    def getLargestRegion(self, thr) :

        imageType = itk.Image[itk.UC, 3]
        filter_  = itk.BinaryShapeKeepNObjectsImageFilter[imageType].New()
        _ = filter_.SetInput(thr)
        _ = filter_.SetAttribute("PhysicalSize")
        _ = filter_.SetNumberOfObjects(int(1))
        _ = filter_.Update()

        return filter_.GetOutput()



    def selectRoi(self, region, image) :

        PixelType, Dim = itk.template(image)[1]
        ImageType = itk.Image[PixelType, Dim]
        roi_extr = itk.RegionOfInterestImageFilter[ImageType,
                                                   ImageType].New()
        _ = roi_extr.SetInput(image)
        _ = roi_extr.SetRegionOfInterest(region)
        _ = roi_extr.Update()

        return roi_extr.GetOutput()

    def computeRois(self) :

        msk1 = None
        msk2 = None

        filter_= median_filter(self.image, 2)
        filtered = execute_pipeline(filter_)
        thr = binary_threshold(filtered, 2500, 0, out_type = itk.UC)
        op = opening(thr, 2)
        thr = execute_pipeline(op)

        largest = self.getLargestRegion(thr)
        bbox = self.getBoundingBox(img = largest)

        start = bbox.GetIndex()
        end = bbox.GetUpperIndex()

        ## define the index for the first image
        self.st1 = [start[0], start[1], start[2]]
        self.ed1 = [int((end[0] + start[0]) / 2), end[1] - 1, end[2] - 1]
        indexes = self._initStartEndIndexes(self.st1, self.ed1)
        region = self._initRegion(indexes)
        im1 = self.selectRoi(region, self.image)

        if self.mask is not None:
            msk1 = self.selectRoi(region, self.mask)

        # extract the second image
        self.st2 = [int((end[0] + start[0])/ 2), start[1], start[2]]
        self.ed2 = end - 1
        indexes = self._initStartEndIndexes(self.st2, self.ed2)
        region = self._initRegion(indexes)
        im2 = self.selectRoi(region, self.image)
        if self.mask is not None :
            msk2 = self.selectRoi(region, self.mask)

        return (im1, msk1), (im2, msk2)


    def reconstruct_image(self, im1, im2, as_mask = False) :

        # FIXME I don't work if mask is not provided
        if as_mask :
            merged_image = np.zeros(self.shape)
        else :
            merged_image = itk.GetArrayFromImage(self.image)

        arr1 = itk.GetArrayFromImage(im1)
        arr2 = itk.GetArrayFromImage(im2)

        merged_image[self.st1[2] : self.ed1[2] + 1, self.st1[1] : self.ed1[1] + 1, self.st1[0] : self.ed1[0] + 1] = arr1
        merged_image[self.st2[2] : self.ed2[2] + 1, self.st2[1] : self.ed2[1] + 1, self.st2[0] : self.ed2[0] + 1] = arr2

        merged_image = array2image(merged_image, self.info)

        return merged_image
