#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itk

from FemurSegmentation.filters import connected_components
from FemurSegmentation.filters import binary_threshold
from FemurSegmentation.filters import region_of_interest
from FemurSegmentation.filters import label_image2shape_label_map
from FemurSegmentation.filters import execute_pipeline
from FemurSegmentation.filters import relabel_components


__author__ = ['Riccardo Biondi']
__email__ = ['riccardo.biondi7@unibo.it']

# Rigurada bene il metodo e uniforma i nomi!! il camel case è necessario


class LegImages :

    def __init__(self, image, mask=None) :

        self.image = image
        self.mask = mask

    def define_regions(self, bbox) :

        RegionType = itk.ImageRegion[3]

        start = bbox.GetIndex()
        end = bbox.GetUpperIndex() - 1
        mid_start = [int((end[0] + start[0]) / 2), start[1], start[2]]
        mid_end = [int((end[0] + start[0] + 1) / 2), end[1], end[2]]

        # region 1
        region1 = RegionType()
        _ = region1.SetIndex(start)
        _ = region1.SetUpperIndex(mid_end)

        # region 2
        region2 = RegionType()
        _ = region2.SetIndex(mid_start)
        _ = region2.SetUpperIndex(end)

        return region1, region2

    def get_legs(self) :

        binarized = binary_threshold(self.image, 3000, -400, out_type=itk.US)
        cc = connected_components(binarized, itk.US)
        cc = execute_pipeline(cc)
        cc = relabel_components(cc)
        pipeline = label_image2shape_label_map(cc)
        label_map = execute_pipeline(pipeline)

        bbox = label_map.GetNthLabelObject(0).GetBoundingBox()

        region1, region2 = self.define_regions(bbox)
        leg1 = execute_pipeline(region_of_interest(self.image, region1))
        leg2 = execute_pipeline(region_of_interest(self.image, region2))

        if self.mask is not None :
            msk1 = execute_pipeline(region_of_interest(self.mask, region1))
            msk2 = execute_pipeline(region_of_interest(self.mask, region2))

            return (leg1, msk1), (leg2, msk2)

        return leg1, leg2

    def reconstruct_image(self) :

        # TODO starting from the two half of the image, reconstruct the original
        # scan
        pass
