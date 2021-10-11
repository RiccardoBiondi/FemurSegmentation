import itk
import numpy as np
import matplotlib.pyplot as plt

from FemurSegmentation.filters import itk_abs
from FemurSegmentation.filters import itk_maximum
from FemurSegmentation.filters import itk_mask_image_filter
from FemurSegmentation.filters import itk_danielsson_distance_map


__author__ = ['Riccardo Biondi']
__email__ = ['riccardo.biondi7@unibo.it']


def itk_label_overlapping_measures(source_image, target_image):

    ImageType = itk.Image[itk.SS, 3]

    measures = itk.LabelOverlapMeasuresImageFilter[ImageType].New()
    _ = measures.SetSourceImage(source_image)
    _ = measures.SetTargetImage(target_image)

    return measures


def itk_hausdorff_distance(source_image, target_image, use_image_spacing=True):

    ImageType = itk.Image[itk.SS, 3]
    hd = itk.HausdorffDistanceImageFilter[ImageType, ImageType].New()

    _ = hd.SetInput1(source_image)
    _ = hd.SetInput2(target_image)
    _ = hd.SetUseImageSpacing(use_image_spacing)

    return hd


def itk_hausdorff_distance_map(source_image, target_image, padding=0):
    '''
    '''

    source2target = itk_distance_map_source_to_target(source_image,
                                                      target_image,
                                                      padding)
    target2source = itk_distance_map_source_to_target(target_image,
                                                      source_image,
                                                      padding)
    _ = source2target.Update()
    _ = target2source.Update()

    hd_map = itk_maximum(source2target.GetOutput(), target2source.GetOutput())

    return hd_map


def itk_distance_map_source_to_target(source_image, target_image, padding=0):
    '''
    '''

    source = cast_image(source_image, itk.F)
    distance = itk_danielsson_distance_map(source)
    _ = distance.Update()
    distance_source2target = itk_mask_image_filter(distance.GetOutput(),
                                                   target_image, masking_value=0,
                                                   out_value=padding)

    return distance_source2target
