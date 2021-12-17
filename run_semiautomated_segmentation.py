#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import itk
import argparse
import numpy as np

from FemurSegmentation.utils import image2array, array2image, cast_image

from FemurSegmentation.filters import add
from FemurSegmentation.filters import erode
from FemurSegmentation.filters import dilate
from FemurSegmentation.filters import binary_threshold
from FemurSegmentation.filters import connected_components
from FemurSegmentation.filters import relabel_components
from FemurSegmentation.filters import execute_pipeline
from FemurSegmentation.filters import binary_curvature_flow
from FemurSegmentation.filters import iterative_hole_filling
from FemurSegmentation.filters import apply_pipeline_slice_by_slice
from FemurSegmentation.filters import itk_invert_intensity
from FemurSegmentation.filters import itk_discrete_gaussian
from FemurSegmentation.filters import itk_binary_morphological_closing
from FemurSegmentation.filters import itk_binary_morphological_opening



from FemurSegmentation.IOManager import ImageReader
from FemurSegmentation.IOManager import VolumeWriter
from FemurSegmentation.boneness import Boneness
from FemurSegmentation.links import GraphCutLinks

try:
    from GraphCutSupport import RunGraphCut

except ModuleNotFoundError:

    lib = {'Linux' : '/lib/',
            'Windows' : r"\lib\\",
            'linux' : '/lib/',
            'windows' : r"\lib\\",
            'ubuntu' : '/lib/',
            'win32' : r"/lib//"}
    #in which OS am I??

    here = os.path.abspath(os.path.dirname(__file__))
    var = ''.join([here, lib[sys.platform]])
    sys.path.append(var)
    from GraphCutSupport import RunGraphCut



__author__ = ['Riccardo Biondi']
__email__ = ['riccardo.biondi7@unibo.it']


def parse_args():
    description = 'Semi-Automated CT Femur Segmentation'
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
    parser.add_argument('--init',
                        dest='init',
                        required=True,
                        type=str,
                        action='store',
                        help='Manual Init of per-pixel term')
    parser.add_argument('--smoothing',
                        dest='smoothing',
                        required=False,
                        type=str,
                        action='store',
                        default=None,
                        help='Specify the kind of smoothing to apply to the final result. if not specified no smoothing will applied. Available: "gaussian", "open_close", "shrink_grow"')
    parser.add_argument('--smooth_size',
                        dest='smooth_size',
                        required=False,
                        action='store',
                        type=int,
                        default=1,
                        help='Kernel size for the smoothing technique')

    args = parser.parse_args()
    return args


def prepare_exclusion_region(image, condition):
    '''
    '''
    ROI = binary_threshold(image, upper_thr=3000, lower_thr=-400)

    cond, info = image2array(condition)

    obj = cond.copy()
    obj[cond == 1] = 1
    obj[cond != 1] = 0

    bkg = cond.copy()
    bkg[cond != 2] = 0
    bkg[cond == 2] = 1

    obj = array2image(obj, info)
    bkg = array2image(bkg, info)

    return ROI, obj, bkg, info


def segment(image, ROI, obj, bkg, info):
    '''
    '''
    bn = Boneness(image=image, scales=[.5, .75], roi=ROI)

    ms_bones = bn.computeBonenessMeasure()

    gc_links = GraphCutLinks(image=image,
                            boneness=ms_bones,
                            roi=ROI,
                            obj=obj,
                            bkg=bkg,
                            sigma=.25,
                            Lambda=100,
                            bone_ms_thr=.3)
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
    labeled = array2image(labelIdImage, info)

    return labeled


def post_processing(labeled):

    # first of all get the largest connected region, to eliminate the spurious
    # points
    labeled = cast_image(labeled, itk.SS)
    cc = execute_pipeline(connected_components(labeled))
    cc = relabel_components(cc)
    lab = binary_threshold(cc, upper_thr=2, lower_thr=0, out_type=itk.F)

    # now apply a surface smoothing
    lab = execute_pipeline(binary_curvature_flow(image=lab))
    lab = binary_threshold(lab, upper_thr=2., lower_thr=.9, out_type=itk.SS)

    # and fill the holes
    lab = iterative_hole_filling(image=lab, max_iter=5, radius=1,
                               majority_threshold=1, bkg_val=0, fgr_val=1)

    return execute_pipeline(lab)
def image_smoothing(image, kind='', kernel=1):

    if kind == 'gaussian':

        print('Applying gaussian smoothing with a kernel size of: {}'.format(kernel), flush=True)
        g = execute_pipeline(itk_discrete_gaussian(image, kernel))

        return g

    elif kind == 'open_close':
        print('Applying open_close smoothing with a kernel size of: {}'.format(kernel), flush=True)

        op = execute_pipeline(itk_binary_morphological_opening(image, radius=kernel))
        cl = execute_pipeline(itk_binary_morphological_closing(op, radius=kernel))

        return cl

    elif kind == 'shrink_grow':

        print('Applying shrink_grow smoothing with a kernel size of: {}'.format(kernel), flush=True)
        er = execute_pipeline(erode(image, radius=kernel))
        di = execute_pipeline(dilate(er, radius=kernel))

        return di

    else:
        return image


def main():

    args = parse_args()

    print('Starting', flush=True)

    print('Reading the input image from:{}'.format(args.input), flush=True)
    print('Reading the initial labeling from: {}'.format(args.init), flush=True)

    reader = ImageReader()
    image = reader(path=args.input, image_type=itk.Image[itk.F, 3])
    init = reader(path=args.init, image_type=itk.Image[itk.SS, 3])

    print('Preparing the initial conditions', flush=True)

    ROI, obj, bkg, info = prepare_exclusion_region(image, init)

    print('Running Graph Cut Segmentation', flush=True)
    labeled = segment(image=image, ROI=ROI, obj=obj, bkg=bkg, info=info)
    print('Refining the Results', flush=True)
    labeled = post_processing(labeled=labeled)


    if args.smoothing is not None:
        if args.smoothing.lower() in ['gaussian', 'open_close', "shrink_grow"]:
            labeled = image_smoothing(labeled, args.smoothing.lower(), args.smooth_size)
        else:
            print('{} is not recognozed as availble smoothing type, no smoothing will be applied'.format(args.smoothing), flush=True)


    print('Writing the results to: {}'.format(args.output), flush=True)
    writer = VolumeWriter(path=args.output, image=labeled)
    _ = writer.write()





    print('[DONE]', flush=True)


if __name__ == '__main__':
    main()
