#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import itk
import argparse
import numpy as np

from FemurSegmentation.utils import image2array
from FemurSegmentation.utils import array2image
from FemurSegmentation.utils import cast_image
from FemurSegmentation.utils import get_optimal_number_of_bins

from FemurSegmentation.IOManager import ImageReader
from FemurSegmentation.IOManager import VolumeWriter

from FemurSegmentation.filters import execute_pipeline
from FemurSegmentation.filters import binary_threshold
from FemurSegmentation.filters import connected_components
from FemurSegmentation.filters import relabel_components
from FemurSegmentation.filters import erode
from FemurSegmentation.filters import binary_curvature_flow
from FemurSegmentation.filters import iterative_hole_filling
from FemurSegmentation.filters import apply_pipeline_slice_by_slice
from FemurSegmentation.filters import distance_map
from FemurSegmentation.filters import add
from FemurSegmentation.filters import median_filter
from FemurSegmentation.filters import adjust_physical_space
from FemurSegmentation.filters import itk_multiple_otsu_threshold
from FemurSegmentation.filters import itk_threshold_below

from FemurSegmentation.boneness import Boneness
from FemurSegmentation.links import GraphCutLinks

try:
    from GraphCutSupport import RunGraphCut

except ModuleNotFoundError:
    here = os.path.abspath(os.path.dirname(__file__))
    var = ''.join([here, r"\lib\\"])
    sys.path.append(var)
    from GraphCutSupport import RunGraphCut

__author__ = ['Riccardo Biondi']
__email__ = ['riccardo.biondi7@unibo.it']


def parse_args():
    description = 'Automated CT Femur Segmentation'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--input',
                        dest='input',
                        required=True,
                        type=str,
                        action='store',
                        help='Input filename')
    parser.add_argument('--output',
                        dest='output',
                        required=True,
                        type=str,
                        action='store',
                        help='Output fileneme')

    args = parser.parse_args()
    return args



def get_optimal_conditions(image, padding_thr=-1500):
    '''
    Determine the optimal condition to use to set the hard constrains for
    the object, background and region of interest

    Parameters
    ----------
    image: itk.Image
        input image to use to compute the conditions
    padding_thr: int
        threhsold to use to remove the padding values
    Returns
    -------
    '''

    # remove the padding value from the image
    noPad = execute_pipeline(itk_threshold_below(image, padding_thr))
    noPad_array = itk.GetArrayFromImage(noPad)

    # find the optimal number of bins
    nob = get_optimal_number_of_bins(noPad_array.reshape(-1))

    # compute multi otsu thresholding
    noPad = cast_image(noPad, itk.SS)
    multi_otsu = itk_multiple_otsu_threshold(noPad, histogram_bins=nob)
    _ = multi_otsu.Update()

    # these thresholds will be used to find the otimal hard constrains
    thr_values = multi_otsu.GetThresholds()
    # get conservative threshold for the obj (mu + 3 * sigma)
    cond = (noPad_array > thr_values[1]) & (noPad_array < thr_values[2])
    mu = np.mean(noPad_array[cond])
    sigma = np.std(noPad_array[cond])

    obj_thr = mu + 3 * sigma
    #return the results

    return [thr_values[0], thr_values[1], thr_values[2], obj_thr]


def get_obj_condition(image, conservative_thr, strict_threshold):
    '''
    Will return the hard constrain based on the original image.
    It will divide the image into two region. In the upper one, containing the
    femur head and the hip-joint, will be applied the strict_threshold, that because
    we want to be sure to exclude all these region between femur head and iliac
    bones.
    In the lowe region, we will apply the conservative_thr, in order to properly
    define a first guess for the cortical bones. here there isn't the problem
    of the hip joint region, so we want to be sure to take as much information
    as possible. The non conservative threshold is applied also to be sure that
    in the knee region s taken also for the subject with a low bone mineral
    density(which have lower HU)
    '''
    PixelType, Dimension = itk.template(image)[1]
    ImageType = itk.Image[PixelType, Dimension]
    # Find the two image region
    imsize = image.GetLargestPossibleRegion().GetSize()
    middle = (4 * imsize[2]) // 5

    RegionType = itk.ImageRegion[3]

    lower_region = RegionType()
    _ = lower_region.SetIndex([0,0,0])
    _ = lower_region.SetUpperIndex([imsize[0] - 1, imsize[1] - 1, middle - 1])

    upper_region = RegionType()
    _ = upper_region.SetIndex([0, 0, middle])
    _ = upper_region.SetUpperIndex([imsize[0] - 1, imsize[1] - 1, imsize[2] - 1])

    ExtractRegionType = itk.RegionOfInterestImageFilter[ImageType, ImageType]

    extract_lower = ExtractRegionType.New()
    _ = extract_lower.SetInput(image)
    _ = extract_lower.SetRegionOfInterest(lower_region)
    _ = extract_lower.Update()
    lower_image = extract_lower.GetOutput()

    extract_upper = ExtractRegionType.New()
    _ = extract_upper.SetInput(image)
    _ = extract_upper.SetRegionOfInterest(upper_region)
    _ = extract_upper.Update()
    upper_image = extract_upper.GetOutput()

    # now compute the binary condition
    lower_array = itk.GetArrayFromImage(lower_image)
    upper_array = itk.GetArrayFromImage(upper_image)

    lcond = lower_array > conservative_thr
    lower_array[lcond] = 1
    lower_array[~lcond] = 0

    ucond = upper_array > strict_threshold
    upper_array[ucond] = 1
    upper_array[~ucond] = 0

    bone_cond = np.concatenate([lower_array, upper_array])

    return bone_cond


def pre_processing(image, #roi_lower_thr=-400,
                #bkg_lower_thr=-400,
                #bkg_upper_thr=-25,
                # to exclude marrow bonesfrom the tissues in per-pixel term
                bkg_bones_low=0.029,
                bkg_bones_up=0.1,
                #obj_thr_gl=600,
                obj_thr_bones=0.3,
                scale=[1.],
                sigma=1.,
                amount=1.,
                thr=0.):

    thresholds = get_optimal_conditions(image)
    ROI = binary_threshold(image, 3000, thresholds[0], out_type=itk.SS)
    ROI = execute_pipeline(erode(ROI, 6))

    # get the largest connected region(Body, will exclude the hospidal bed and
    # CT tube from the region of interest)
    ROI = execute_pipeline(connected_components(ROI))
    ROI = relabel_components(ROI)
    ROI = binary_threshold(ROI, 2, 0, out_type=itk.UC)

    # now compute the boneness measure. Will be used to dtermine the bkg and
    # obj exclusion region
    bones = Boneness(image, scale, ROI)
    boneness = bones.computeBonenessMeasure()
    boneness, _ = image2array(boneness)


    obj_gl_cond = get_obj_condition(image, thresholds[3], thresholds[2])
    obj, info = image2array(image)
    cond = (obj_gl_cond == 1) & (boneness > obj_thr_bones)
    obj[cond] = 1
    obj[~cond] = 0

    obj = array2image(obj, info)

    bkg, info = image2array(image)
    cond = (bkg > thresholds[0]) & (bkg < thresholds[1]) & (boneness < bkg_bones_up)
    bkg[cond] = 1
    bkg[~cond] = 0
    bkg = array2image(bkg, info)

    # now get the largest connected region, which will be(hopely) the
    # femur region
    obj = cast_image(obj, itk.SS)
    obj = median_filter(obj, 1)
    cc = connected_components(obj.GetOutput())
    cc_im = execute_pipeline(cc)
    rel = relabel_components(cc_im)
    obj = binary_threshold(rel, 2, 0, out_type=itk.UC)

    boneness = array2image(boneness, info)

    return image, ROI, bkg, obj


def segment(image, obj, bkg, ROI, scales=[.5, .75], sigma=.25,
            Lambda=100, bone_ms_thr=.3):

    _, info = image2array(image)
    # compute multiscale boneness
    bones = Boneness(image, scales, ROI)
    boneness = bones.computeBonenessMeasure()
    # compute links
    gc_links = GraphCutLinks(image, boneness, ROI, obj, bkg, sigma=sigma,
                            Lambda=Lambda, bone_ms_thr=bone_ms_thr)
    cost_sink_flatten, cost_source_flatten, cost_vx, CentersVx, NeighborsVx, _totalNeighbors, costFromCenter, costToCenter = gc_links.getLinks()
        # apply graph cut
    uint_gcresult = RunGraphCut(gc_links.total_vx,
                                np.ascontiguousarray(cost_vx, dtype=np.uint32),
                                np.ascontiguousarray(cost_source_flatten, dtype=np.uint32),
                                np.ascontiguousarray(cost_sink_flatten, dtype=np.uint32),
                                _totalNeighbors,
                                np.ascontiguousarray(CentersVx, dtype=np.uint32),
                                np.ascontiguousarray(NeighborsVx, dtype=np.uint32),
                                np.ascontiguousarray(costFromCenter, dtype=np.uint32),
                                np.ascontiguousarray(costToCenter, dtype=np.uint32))

    labelIdImage = gc_links.vx_id
    labelIdImage[gc_links.vx_id != -1] = uint_gcresult
    labelIdImage[gc_links.vx_id == -1] = 0
    labelIdImage = np.asarray(labelIdImage, dtype=np.uint8)
    labeled = array2image(labelIdImage, info)

    return labeled


def post_processing(labeled):

    labeled = cast_image(labeled, itk.F)
    f = binary_curvature_flow(labeled, number_of_iterations=5)
    lab = execute_pipeline(f)
    lab = binary_threshold(lab, upper_thr=2., lower_thr=.9, out_type=itk.SS)

    filled = cast_image(lab, itk.US)
    cc = connected_components(filled, itk.US)
    cc_im = execute_pipeline(cc)
    rel = relabel_components(cc_im)
    filled = binary_threshold(rel, 2, 0, out_type=itk.UC)

    filler = iterative_hole_filling(filled, max_iter=5, radius=1)
    pipe = distance_map(filler.GetOutput())
    dist = execute_pipeline(pipe)
    dist = binary_threshold(dist, 25, 0, out_type=itk.SS)

    negative = itk.ConnectedComponentImageFilter[itk.Image[itk.SS, 2],
                                                itk.Image[itk.SS, 2]].New()

    negative = apply_pipeline_slice_by_slice(dist, negative)
    negative = execute_pipeline(negative)
    negative = binary_threshold(negative, 700, 1)

    filled = add(filled, negative)

    filler = iterative_hole_filling(filled, max_iter=10, radius=1)
    filled = execute_pipeline(filler)

    filled = cast_image(filled, itk.US)
    med = median_filter(image=filled, radius=1)
    cc = connected_components(med.GetOutput(), itk.US)
    cc_im = execute_pipeline(cc)
    rel = relabel_components(cc_im)
    filled = binary_threshold(rel, 2, 0, out_type=itk.UC)

    return filled


def segmentation_refinement(image, obj, roi):
    '''
    Perform a secon graph cut using the previous estimated bones to set the hard
    constrains. This allows to fill all the holes in the femur head and knee.
    This allows to refine the segmentation, make it suitable for the Finite Element
    Model.

    Parameters
    ----------
    image: itk.Image
        original CT scan
    obj: itk.Image
        bone segmentation results
    roi: itk.Image
        Region to Focus the graph cut: allows to reduce the algorithm complexity
    Returns
    -------
    femur: itk.Image
        final segmentation results
    '''
    _, info = image2array(image)
    bkg = binary_threshold(image, upper_thr=0, lower_thr=-100, out_type=itk.UC)

    bones = Boneness(image, [0.75], roi)
    boneness = bones.computeBonenessMeasure()

    gc_links = GraphCutLinks(image, boneness, roi, obj, bkg, sigma=0.5,
                            Lambda=1000, bone_ms_thr=0.0)

    cost_sink_flatten, cost_source_flatten, cost_vx, CentersVx, NeighborsVx, _totalNeighbors, costFromCenter, costToCenter = gc_links.getLinks()

    uint_gcresult = RunGraphCut(gc_links.total_vx,
                                np.ascontiguousarray(cost_vx, dtype=np.uint32),
                                np.ascontiguousarray(cost_source_flatten, dtype=np.uint32),
                                np.ascontiguousarray(cost_sink_flatten, dtype=np.uint32),
                                _totalNeighbors,
                                np.ascontiguousarray(CentersVx, dtype=np.uint32),
                                np.ascontiguousarray(NeighborsVx, dtype=np.uint32),
                                np.ascontiguousarray(costFromCenter, dtype=np.uint32),
                                np.ascontiguousarray(costToCenter, dtype=np.uint32))

    labelIdImage = gc_links.vx_id
    labelIdImage[gc_links.vx_id != -1] = uint_gcresult
    labelIdImage[gc_links.vx_id == -1] = 0
    labelIdImage = np.asarray(labelIdImage, dtype=np.uint8)

    label = array2image(labelIdImage, info)

    return label


def final_refinement_and_filling(image):
    '''
    This pipeline will perform the final filling and last refinement for the
    refined graph cut
    '''
    # invert the image-> we want to find the complementary because we want to
    # fill the image
    PixelType, Dimension = itk.template(image)[1]
    ImageType = itk.Image[PixelType, Dimension]
    inverter = itk.InvertIntensityImageFilter[ImageType, ImageType].New()
    _ = inverter.SetInput(image)
    _ = inverter.SetMaximum(1)
    _ = inverter.Update()
    inverted = inverter.GetOutput()

    #now, slice by slice, take all the connected components except the largest
    # one, which is the background(remember that I have inverted the image)
    negative = itk.ConnectedComponentImageFilter[itk.Image[itk.UC, 2],
                                                itk.Image[itk.UC, 2]].New()

    negative = apply_pipeline_slice_by_slice(inverted, negative)
    negative = execute_pipeline(negative)
    negative = binary_threshold(negative, 700, 1)

    # now sum up the complementary and the image to refine
    refined = add(image, negative)

    # and finally take the largest connected region -> will remove all the
    # spurious regions
    cc = itk.ConnectedComponentImageFilter[itk.Image[itk.UC, 3], itk.Image[itk.UC, 3]].New()
    _ = cc.SetInput(refined)
    _ = cc.Update()
    rel = relabel_components(cc.GetOutput())
    refined = binary_threshold(rel, upper_thr=2, lower_thr=0)

    return refined


def main():

    # parse the arguments: Input image, Output Destination
    args = parse_args()

    # read the image to process
    reader = ImageReader(path=args.input, image_type=itk.Image[itk.F, 3])
    image = reader.read()

    image, ROI, bkg, obj = pre_processing(image=image)
    labeled = segment(image=image, obj=obj, bkg=bkg, ROI=ROI)
    labeled = post_processing(labeled=labeled)

    labeled = adjust_physical_space(in_image=labeled,
                                    ref_image=image,
                                    ImageType=itk.Image[itk.UC, 3])

    labeled = segmentation_refinement(image, labeled, ROI)

    labeled = final_refinement_and_filling(labeled)

    writer = VolumeWriter(path=args.output, image=labeled)
    _ = writer.write()


if __name__ == '__main__':
    main()
