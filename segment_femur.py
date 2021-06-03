import os
import itk
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt

# third part
from FemurSegmentation.IOManager import ImageReader, VolumeWriter

from FemurSegmentation.utils import image2array, array2image, cast_image
from FemurSegmentation.utils import get_labeled_leg

from FemurSegmentation.filters import binary_threshold
from FemurSegmentation.filters import connected_components
from FemurSegmentation.filters import relabel_components
from FemurSegmentation.filters import execute_pipeline
from FemurSegmentation.filters import iterative_hole_filling
from FemurSegmentation.filters import connected_filter_slice_by_slice
from FemurSegmentation.filters import distance_map
from FemurSegmentation.filters import add

from FemurSegmentation.image_splitter import LegImages
from FemurSegmentation.boneness import Boneness
from FemurSegmentation.links import GraphCutLinks

from GraphCutSupport import RunGraphCut

def view(image, idx) :

    arr = itk.GetArrayFromImage(image)

    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 10))
    _ = ax.axis('off')
    _ = ax.imshow(arr[idx], cmap = 'gray')

    plt.show()

def parse_args() :
    description = 'A GraphCut based framework for the femur segmentation'
    parser = argparse.ArgumentParser(description = description)

    _ = parser.add_argument('--input',
                            dest='input',
                            required=True,
                            type=str,
                            action='store',
                            help='Path to the input image')

    _ = parser.add_argument('--output',
                            dest='output',
                            required=True,
                            type=str,
                            action='store',
                            help='path to the output folder')

    _ = parser.add_argument('--mask_path',
                            dest='mask_path',
                            required=False,
                            type=str,
                            default='')
    args = parser.parse_args()


    return args



def pre_processing(image) :
    '''
    Pre process the image and estimate the ROI in which compute the graph,
    the object and background voxels to set the Source and Sink tLinks
    '''
    print("\tI am pre-processing...", flush = True)

    # find the ROI
    ROI = binary_threshold(image, 3000, -100, out_type = itk.UC)

    # compute the boneness measure
    bones = Boneness(image, [1.], ROI)
    boneness = bones.computeBonenessMeasure()
    boneness, _ = image2array(boneness)

    # find background
    # TODO Improve background condition
    bkg = binary_threshold(image, -25, -100, out_type = itk.UC)

    # to find the object it will combine the threshold information, which select
    # only the bones region, the boneness measure which allows to separate the
    # joint, in the end takes the largest connected component to consider only
    # the femur
    obj, info = image2array(image)

    cond = (obj > 600) & (boneness > 0.3)
    obj[cond] = 1
    obj[~cond] = 0

    obj = array2image(obj, info)
    obj = cast_image(obj, itk.SS)
    cc = connected_components(obj)
    cc_im = execute_pipeline(cc)
    rel = relabel_components(cc_im)
    obj = binary_threshold(rel, 2, 0, out_type = itk.UC)


    # TODO add unsharp mask

    # return unsharp masked image, ROI, obj and bkg

    return ROI, bkg, obj



def segmentation(image, obj, bkg, ROI) :

    _, info = image2array(image)
    # compute multiscale boneness
    bones = Boneness(image, [0.5, 1.], ROI)
    boneness = bones.computeBonenessMeasure()
    # compute links
    gc_links = GraphCutLinks(image, boneness, ROI, obj, bkg)
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
                                              np.ascontiguousarray(costToCenter, dtype=np.uint32)
                                              )

    labelIdImage = gc_links.vx_id
    labelIdImage[gc_links.vx_id != -1] = uint_gcresult
    labelIdImage[gc_links.vx_id == -1] = 0
    labelIdImage = np.asarray(labelIdImage, dtype=np.uint8)
    labeled = array2image(labelIdImage, info)

    return labeled




def post_processing(labeled) :
    '''
    Fill the labeled image. Get only the femur
    '''

    # get the largest component (femur)
    filled = cast_image(labeled, itk.US)
    cc =  connected_components(filled, itk.US)
    cc_im = execute_pipeline(cc)
    rel = relabel_components(cc_im)
    filled = binary_threshold(rel, 2, 0, out_type = itk.UC)

    filler = iterative_hole_filling(filled, max_iter = 5, radius = 3)
    pipe = distance_map(filler.GetOutput())
    dist = execute_pipeline(pipe)

    negative = connected_filter_slice_by_slice(dist)
    negative = binary_threshold(negative, 700, 1)

    filled = add(filled, negative)

    filler = iterative_hole_filling(filled, max_iter = 5, radius = 3)
    filled = execute_pipeline(filler)

    filled = cast_image(filled, itk.US)
    cc = connected_components(filled, itk.US)
    cc_im = execute_pipeline(cc)
    rel = relabel_components(cc_im)
    filled = binary_threshold(rel, 2, 0, out_type = itk.UC)

    return filled



def main(image) :

    ROI, bkg, obj = pre_processing(image)
    labeled = segmentation(image, obj, bkg, ROI)
    labeled = post_processing(labeled)

    return labeled


if __name__ == '__main__' :

    args = parse_args()

    print('I am reading the image from: {}'.format(args.input), flush=True)
    reader = ImageReader(args.input, itk.Image[itk.F, 3])
    image =reader.read()


    # this part is used because the dataset we are used has only one labeled
    # leg. This allow us to discriminate between the labeled and unlabeled one
    # and process only one part of the image, reducing the computational time.
    if args.mask_path != '' :

        print('Mask Sepcified, I am reading the GT mask from: {}'.format(args.mask_path), flush=True)
        reader = ImageReader(args.mask_path, itk.Image[itk.UC, 3])
        mask = reader.read()

        splitter = LegImages(image, mask)

        leg1, leg2 = splitter.computeRois()

        leg, _ = get_labeled_leg(leg1, leg2)

    label = main(image)


    # now reconstruct the original image
