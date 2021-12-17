#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import itk
import argparse
import platform
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm, iqr
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit


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
from FemurSegmentation.filters import dilate
from FemurSegmentation.filters import apply_pipeline_slice_by_slice
from FemurSegmentation.filters import add
from FemurSegmentation.filters import median_filter
from FemurSegmentation.filters import adjust_physical_space
from FemurSegmentation.filters import itk_multiple_otsu_threshold
from FemurSegmentation.filters import itk_threshold_below
from FemurSegmentation.filters import itk_binary_morphological_closing
from FemurSegmentation.filters import itk_binary_morphological_opening
from FemurSegmentation.filters import itk_invert_intensity
from FemurSegmentation.filters import itk_otsu_threshold
from FemurSegmentation.filters import region_of_interest
from FemurSegmentation.filters import itk_sigmoid
from FemurSegmentation.filters import itk_geodesic_active_contour
from FemurSegmentation.filters import itk_signed_maurer_distance_map
from FemurSegmentation.filters import unsharp_mask

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
    description = 'Automated CT Femur Segmentation'
    parser = argparse.ArgumentParser(description=description)

    #parser.add_argument('--input',
    #                    dest='input',
    #                    required=True,
    #                    type=str,
    #                    action='store',
    #                    help='Input filename')
    parser.add_argument('--left',
                        dest='left',
                        required=True,
                        type=str,
                        action='store',
                        help='Left Leg Input')
    parser.add_argument('--right',
                        dest='right',
                        required=True,
                        type=str,
                        action='store',
                        help='Right Leg Input')
    parser.add_argument('--output',
                        dest='output',
                        required=True,
                        type=str,
                        action='store',
                        help='Output fileneme')

    args = parser.parse_args()
    return args


def itk_curvature_flow(image, number_of_iteration,time_step):

    PixelType, Dimension = itk.template(image)[1]
    ImageType = itk.Image[PixelType, Dimension]

    filter_ = itk.CurvatureFlowImageFilter[ImageType, ImageType].New()
    _ = filter_.SetInput(image)
    _ = filter_.SetNumberOfIterations(number_of_iteration)
    _ = filter_.SetTimeStep(time_step)

    return filter_


def itk_normalize(image, mask):

    ImagePixelType, Dimension = itk.template(image)[1]
    MaskPixelType, Dimension = itk.template(mask)[1]

    ImageType = itk.Image[ImagePixelType, Dimension]
    MaskType = itk.Image[MaskPixelType, Dimension]

    filter_ = itk.LabelStatisticsImageFilter[ImageType, MaskType].New()
    _ = filter_.SetInput(image)
    _ = filter_.SetLabelInput(mask)
    _ = filter_.Update()

    mu = filter_.GetMean(1)
    sigma = filter_.GetSigma(1)

    normalized = itk_shift_and_scale(image, mu, 1 / sigma)


    return normalized


def itk_shift_and_scale(image, shift=0, scale=1):
    PixelType, Dimension = itk.template(image)[1]
    ImageType = itk.Image[PixelType, Dimension]
    OutputType = itk.Image[itk.F, 3]

    filter_ = itk.ShiftScaleImageFilter[ImageType, OutputType].New()
    _ = filter_.SetShift(shift)
    _ = filter_.SetScale(scale)
    _ = filter_.SetInput(image)

    return filter_


def itk_subtract(image1, image2):

    PixelType, Dimension = itk.template(image1)[1]
    ImageType = itk.Image[PixelType, Dimension]

    difference = itk.SubtractImageFilter[ImageType, ImageType, ImageType].New()
    _ = difference.SetInput1(image1)
    _ = difference.SetInput2(image2)


    return difference


def flood_fill_2d(image):
    '''
    '''
    inverted = itk_invert_intensity(image)

    # flood fill
    negative = itk.ConnectedComponentImageFilter[itk.Image[itk.SS, 2], itk.Image[itk.SS, 2]].New()

    negative = apply_pipeline_slice_by_slice(inverted.GetOutput(), negative)
    negative = execute_pipeline(negative)
    negative = binary_threshold(negative, 700, 1)

    flood_filled = add(image, negative)
    flood_filled = cast_image(flood_filled, itk.SS)

    return flood_filled


def find_femur_limit(image, thr):

    cortical_bones = binary_threshold(image, lower_thr=thr, upper_thr=10000, out_type=itk.SS)
    closed = itk_binary_morphological_closing(cortical_bones, radius=6)

    median = execute_pipeline(median_filter(closed.GetOutput()))
    bones = flood_fill_2d(median)

    eroded = execute_pipeline(erode(bones, radius=7))

    mask, info = image2array(eroded)
    im, info = image2array(image)
    im[mask == 0] = -3000 # set all the values outside the bone region as padding values
    masked  = array2image(im, info)


    p = binary_threshold(masked, lower_thr=thr, upper_thr=2000)
    p = execute_pipeline(median_filter(p, 1))

    p_array = itk.GetArrayFromImage(p)

    p_vol = np.asarray([np.sum(v) for v in p_array])
    p_grad = np.gradient(p_vol)

    max_slice = np.argmin(p_grad[len(p_grad) // 2 : ]) + (len(p_grad) // 2)
    min_slice = 3 * (len(p_vol) // 4)


    print('Upper bound: {}'.format(max_slice))
    print('Lower Bound: {}'.format(min_slice))

    return min_slice, max_slice, p



def get_air_peack(image):

    n = itk.GetArrayFromImage(image)

    hist, bin = np.histogram(n[n > -10], bins='fd')

    peack = bin[np.argmax(hist)+1]


    range_ = [peack - 1, peack + 1]

    cond = (n > range_[0]) & ( n < range_[1])
    sigma = np.std(n[cond])

    return peack, sigma


def get_optimal_obj_stats(image, body):

    b = itk.GetArrayFromImage(body)
    n = itk.GetArrayFromImage(image)
    n[b != 1] = -3000

    mean = np.mean(n[n != -3000])
    std = np.std(n[n != -3000])

    return mean, std



class NewGraphCutLinks :

    def __init__(self,
                image,
                boneness,
                roi,
                obj,
                bkg,
                sigma=.25,
                Lambda=50.,
                bone_ms_thr=0.2) :
        '''
        '''

        # TODO add input controls
        self.image, _ = image2array(image)
        self.boneness, _ = image2array(boneness)
        self.roi, _ = image2array(roi)
        self.bkg, _ = image2array(bkg)
        self.obj, _ = image2array(obj)
        self.total_vx = int(np.sum(self.roi))
        self.vx_id = np.full(self.image.shape, -1)
        self.vx_id[self.roi != 0] = range(self.total_vx)

        self.sigma = sigma
        self.Lambda = float(Lambda)
        self.bone_ms_thr = bone_ms_thr

    def bonenessCost(self, vx_left, vx_right, sh_left, sh_right) :
        '''
        '''
        # both voxel must be in ROI
        cond = (vx_left > -1) & (vx_right > -1)

        from_center = np.full(vx_left[cond].shape, self.Lambda)
        to_center = np.full(vx_left[cond].shape, self.Lambda)

        den = 2. * (self.sigma ** 2)
        num = np.abs(sh_left[cond] - sh_right[cond])**2

        # compute cost from center
        cond_a = sh_left[cond] > sh_right[cond]
        from_center[cond_a] *= np.exp(- num[cond_a] / den)

        # compute cost to center
        cond_b = sh_left[cond] < sh_right[cond]
        to_center[cond_b] *= np.exp(- num[cond_b] / den)

        return (vx_left[cond], vx_right[cond]), to_center, from_center

    def tLinkSource(self) :
        cost_source = self.Lambda * self.bkg

        return cost_source

    def tLinkSink(self) :

        cost_sink = self.Lambda * self.obj

        return cost_sink

    def nLinks(self) :

        X, Xto, Xfrom = self.bonenessCost(self.vx_id[:-1, :, :],
                                          self.vx_id[1:, :, :],
                                          self.boneness[:-1, :, :],
                                          self.boneness[1:, :, :])

        Y, Yto, Yfrom = self.bonenessCost(self.vx_id[:, :-1, :],
                                          self.vx_id[:, 1:, :],
                                          self.boneness[:, :-1, :],
                                          self.boneness[:, 1:, :])

        Z, Zto, Zfrom = self.bonenessCost(self.vx_id[:, :, :-1],
                                          self.vx_id[:, :, 1:],
                                          self.boneness[:, :, :-1],
                                          self.boneness[:, :, 1:])

        CentersVx = np.concatenate([Z[0], Y[0], X[0]])
        NeighborsVx = np.concatenate([Z[1], Y[1], X[1]])
        _totalNeighbors = len(NeighborsVx)
        costFromCenter = np.concatenate([Zfrom, Yfrom, Xfrom])
        costToCenter = np.concatenate([Zto, Yto, Xto])

        return CentersVx, NeighborsVx, _totalNeighbors, costFromCenter, costToCenter

    def getLinks(self) :

        # get tLinks
        source = self.tLinkSource()
        sink = self.tLinkSink()
        # flatten tLinks
        cost_sink_flatten = sink[self.vx_id != -1]
        cost_source_flatten = source[self.vx_id != -1]
        cost_vx = self.vx_id[self.vx_id != -1]

        # nLinks
        CentersVx, NeighborsVx, _totalNeighbors, costFromCenter, costToCenter = self.nLinks()

        return cost_sink_flatten, cost_source_flatten, cost_vx, CentersVx, NeighborsVx, _totalNeighbors, costFromCenter, costToCenter


def get_largest_component(image):

    image = cast_image(image, itk.SS)
    cc = execute_pipeline(connected_components(image))
    rel = relabel_components(cc)
    out = binary_threshold(rel, lower_thr=0, upper_thr=2)

    return out


def get_knee_slice(image):

    im_array = itk.GetArrayFromImage(image)
    im_vol = np.asarray([np.sum(v) for v in im_array])
    im_grad = np.gradient(im_vol)

    idx = np.argmax(im_grad[ : len(im_grad) // 3])

    while (im_grad[idx] > 0.) & (idx > -1):
        idx = idx - 1
    return idx + 1


        # ██████  ██████  ███████     ██████  ██████   ██████   ██████ ███████ ███████ ███████ ██ ███    ██  ██████
        # ██   ██ ██   ██ ██          ██   ██ ██   ██ ██    ██ ██      ██      ██      ██      ██ ████   ██ ██
        # ██████  ██████  █████ █████ ██████  ██████  ██    ██ ██      █████   ███████ ███████ ██ ██ ██  ██ ██   ███
        # ██      ██   ██ ██          ██      ██   ██ ██    ██ ██      ██           ██      ██ ██ ██  ██ ██ ██    ██
        # ██      ██   ██ ███████     ██      ██   ██  ██████   ██████ ███████ ███████ ███████ ██ ██   ████  ██████

def get_object_conditions(image, p, thr, boneness, min_slice, max_slice):

    cortical = binary_threshold(image, lower_thr=thr, upper_thr=100., out_type=itk.SS)

    cortical = flood_fill_2d(cortical)

    cortical = cast_image(cortical, itk.SS)
    cortical = execute_pipeline(median_filter(cortical, 1))

    cortical, info = image2array(cortical)
    cortical[min_slice: ] = 0

    cortical = array2image(cortical, info)

    cc = connected_components(cortical)
    cc_im = execute_pipeline(cc)
    rel = relabel_components(cc_im)
    cortical = binary_threshold(rel, 2, 0, out_type=itk.UC)


    mask = cast_image(p, itk.SS)
    close = execute_pipeline(itk_binary_morphological_closing(mask, 2))
    open = execute_pipeline(itk_binary_morphological_opening(mask, 2))
    close = cast_image(close, itk.SS)
    mask = cast_image(mask, itk.SS)


    diff = execute_pipeline(itk_subtract(close, open))

    eroded = execute_pipeline(erode(diff, 1))

    cc = execute_pipeline(connected_components(eroded))
    rel = relabel_components(cc)
    head = binary_threshold(rel, lower_thr=0, upper_thr=2)

    cortical, info = image2array(cortical)
    head, _ = image2array(eroded)

    head[max_slice: ] = 0
    head [:min_slice] = 0
    cond = (head == 1) & (boneness > .1)
    head[cond] = 1
    head[~cond] = 0


    obj = cortical + head
    obj = array2image(obj, info)

    return obj


def get_bkg_conditions(image, boneness, min_slice, max_slice):

    bkg, info = image2array(image)
    im = bkg.copy()

    cond = (bkg < 1.) & (boneness < -0.25)# & (boneness > -.5)

    bkg[cond] = 1
    bkg[~cond] = 0

    bkg[max_slice:] = (im[max_slice: ] > 0).astype(np.uint8)


    bkg = array2image(bkg, info)

    return bkg



def pre_processing(image, body, gamma1=6.3, gamma2=2.2):

    print('\tNormalize the image', flush=True)

    normalized = execute_pipeline(itk_normalize(image, body))
    mu, sigma = get_air_peack(normalized)

    print('\tunsharping', flush=True)

    unsharped = execute_pipeline(unsharp_mask(normalized, sigma=gamma1 * sigma,
                                              amount=1.0, threhsold=0.0))
    mean, std = get_optimal_obj_stats(unsharped, body)
    obj_thr = mean +  gamma2 * std

    print('\tFind femur limit', flush=True)
    min_slice, max_slice, p = find_femur_limit(unsharped, obj_thr)

    print('\t computing boneness measure', flush=True)

    bones = Boneness(unsharped, [.6, .8], body)
    boneness = bones.computeBonenessMeasure()
    boneness, info = image2array(boneness)

    print('\t get object', flush=True)

    obj = get_object_conditions(image=unsharped, p=p, thr=obj_thr,
                                boneness=boneness, min_slice=min_slice,
                                max_slice=max_slice)

    print('\t get background', flush=True)
    bkg = get_bkg_conditions(image=unsharped, boneness=boneness,
                             min_slice=min_slice, max_slice=max_slice)

    boneness = array2image(boneness, info)

    return obj, bkg, boneness



def graph_cut(gc_links, info):


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

    labeled = cast_image(labeled, itk.SS)

    closed = itk_binary_morphological_closing(labeled, radius=1)
    median = execute_pipeline(median_filter(closed.GetOutput()))

    l = flood_fill_2d(median)

    return l


def run_segmentation(in_path, out_path ):

    print('Start Segmentation', flush=True)

    writer = VolumeWriter()
    reader = ImageReader()
    print('Reading Image from: {}'.format(in_path), flush=True)
    image = reader(in_path, itk.Image[itk.F, 3])

    _, info = image2array(image)
    print('Pre-Processing', flush=True)

    # find the body region, which is the one of interest
    body = binary_threshold(image, 3000, -400, out_type=itk.SS)
    body = execute_pipeline(erode(body, 6))
    body = execute_pipeline(connected_components(body))
    body = relabel_components(body)
    body = binary_threshold(body, 2, 0, out_type=itk.UC)

    obj, bkg, boneness = pre_processing(image, body)

    print('pre-segmentating', flush=True)
    #
    # Init graph cut links
    #
    gc_links = GraphCutLinks(image, boneness, body, obj, bkg, sigma=.25,
                            Lambda=100, bone_ms_thr=.7)

    labeled = graph_cut(gc_links, info)
    labeled = post_processing(labeled)
    labeled = get_largest_component(labeled)

    #_ = writer(intermediate_path, labeled)

    #
    # Now prepare the new obj and bkg conditions
    #
    knee_slice = get_knee_slice(labeled)
    l, info = image2array(labeled)
    l[:knee_slice + 2] = 0
    labeled = array2image(l, info)

    print('Segmenting', flush=True)

    new_obj = execute_pipeline(erode(labeled, 3))
    new_bkg = execute_pipeline(dilate(labeled, 2))

    new_bkg = execute_pipeline(itk_invert_intensity(new_bkg))

    new_obj = execute_pipeline(itk_signed_maurer_distance_map(new_obj))
    new_bkg = execute_pipeline(itk_signed_maurer_distance_map(new_bkg))

    new_obj = execute_pipeline(itk_curvature_flow(new_obj, 2,0.625))
    new_bkg = execute_pipeline(itk_curvature_flow(new_bkg, 2,0.625))

    gc_links = NewGraphCutLinks(image, boneness, body, new_obj, new_bkg, sigma=.25,
                            Lambda=100, bone_ms_thr=.7)
    labeled = graph_cut(gc_links, info)
    labeled = get_largest_component(labeled)


    print('Writing the result to: {}'.format(out_path), flush=True)

    _ = writer(out_path , labeled)



def main():

    args = parse_args()
    # get names
    #names = os.listdir(args.input)
    #to_remove = ['D0012', 'D0020', 'D0032', 'D0038', 'D0037', 'D0039', 'D0040', 'D0047', 'D0048', 'D0049', 'D0054', 'D0062', 'D0068', 'D0277']
    #names = list(filter(lambda x : 'D0' in x, names))
    #names = list(filter(lambda x: int(x[1:]) > 223, names))

    left_out = '{}_seg_L.nrrd'.format(args.output)
    right_out = '{}_seg_R.nrrd'.format(args.output)

    #for name in names:
    #    print('I am processing: {}'.format(name), flush=True)

    #    in_path = '{base}/{pat}/{pat}_im.nrrd'.format(base=args.input, pat=name)
    #    out_path = '{out}/{pat}.nrrd'.format(out=args.output, pat=name)
    #    inter_path = '{inter}/{pat}.nrrd'.format(inter=args.intermediate, pat=name)
    print('Segmenting the Left Leg', flush=True)
    _ = run_segmentation(args.left, left_out)
    print('Segmenting the Right Leg', flush=True)
    _ = run_segmentation(args.right, right_out)
    print('[DONE]', flush=True)




if __name__ == '__main__':
    main()
