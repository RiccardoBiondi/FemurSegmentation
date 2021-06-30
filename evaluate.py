#!/bin/env python

import os
import itk
import argparse
import numpy as np
import pandas as pd
from glob import glob

from FemurSegmentation.utils import image2array, array2image
from FemurSegmentation.IOManager import ImageReader
from FemurSegmentation.filters import execute_pipeline
from FemurSegmentation.metrics import dice_score, housdorff_distance
from FemurSegmentation.metrics import danielsson_distance_map


df_filename = '/mnt/d/Riccardo/Progetti/Aperti/FemurSegmentation/Data/evaluate.csv'
mask_path = '/mnt/d/Riccardo/Progetti/Aperti/FemurSegmentation/Data/Preliminary_gc_results/{}_gc.nrrd'
gt_path = '/mnt/d/Riccardo/Progetti/Aperti/FemurSegmentation/Data/Formatted/{}/{}_gt.nrrd'


def get_names(path, split_condition='_'):
    '''
    Sample names to evaluate

    Parameters
    ----------
    path: str
        path to the folder in which samples to evaluate are stored

    Return
    ------
    list: list of str
        list with the basename of the samples
    '''
    sample_paths = sorted(glob(path.format('*')))
    names = list(map(lambda x : os.path.basename(x).split(split_condition)[0], sample_paths))

    return names

def parse_args():
    pass


def get_hip_joint_region(image1, image2):
    '''
    Since the segmentation in the Femur body is usually perfect, this will
    improve the scores even if the segmentation in the hip joint region
    (the one in which we are interested in), is not so good.

    This function aims to return only this region, in order to evaluate the
    scores only here.

    Parameters
    ----------
    y_pred: itk.Image
        Segmentation predicted by the model
    y_true: itk.Image
        grount truth segmentation

    Return
    ------
    pred_hip_joint: np.ndarray
        image array containing the upper 1/3 of y_pred
    true_hip_joint: np.ndarray
        image containing the upper 1/3 of y_true
    '''

    p_array, info = image2array(y_pred)
    t_array, _ = image2array(y_true)

    p_shape = p_array.shape
    t_shape = t_array.shape

    assert p_shape == t_shape

    p_array = p_array[2 * p_shape[0] // 3 :]
    t_array = t_array[2 * t_shape[0] // 3 :]

    info['Size'] = p_array.shape
    info['Upper Index'] = p_array.shape


    return array2image(p_array, info), array2image(t_array, info)


def housdorff_distance_slice_by_slice(image1, image2,
                                      mode='HD', use_image_spacing=False):

    dX = danielsson_distance_map(image1, use_image_spacing=use_image_spacing)
    dY = danielsson_distance_map(image2, use_image_spacing=use_image_spacing)

    dX = execute_pipeline(dX)
    dY = execute_pipeline(dY)

    X, _ = image2array(image1)
    dX, _ = image2array(dX)
    Y, _ = image2array(image2)
    dY, _ = image2array(dY)

    dX[Y==0] = 0
    dY[X==0] = 0
    hXY = np.max(dX, axis=(1, 2))
    hYX = np.max(dY, axis=(1, 2))

    return np.max([hXY, hYX], axis=0)


def evaluate_slice_by_slice(image1, image2):
    '''
    '''
    im1, _ = image2array(image1)
    im2, _ = image2array(image2)
    scores = [dice_score(i1, i2) for i1, i2 in zip(im1, im2)]

    return scores


def evaluate(image1, image2):

    # overall
    overall_dice = dice_score(image1, image2)
    overall_hd = housdorff_distance(image1, image2, use_image_spacing=True)
    # profile
    dice_profile = evaluate_slice_by_slice(image1, image2)
    hd_profile = housdorff_distance_slice_by_slice(image1, image2,
                                                   use_image_spacing=True)

    # only hip joint
    hip1, hip2 = get_hip_joint_region(image1, image2)

    hip_dice = dice_score(hip1, hip2)
    hip_hd = housdorff_distance(hip1, hip2, use_image_spacing=True)

    return {'overall dice' : overall_dice,
            'overall hd' : overall_hd,
            'hip dice' : hip_dice,
            'hip hd' : hip_hd,
            'dice profile' : [dice_profile],
            'hd profile' : [hd_profile]}


def main(name, y_pred, y_true):

    evaluation = evaluate(y_pred, y_true)

    df = pd.DataFrame(evaluation)
    df['Patient'] = name

    return df


if __name__ == '__main__':

    names = get_names(path=mask_path)

    for name in names:
        print('I am processing {}'.format(name), flush=True)

        reader = ImageReader(path=mask_path.format(name),
                            image_type=itk.Image[itk.UC, 3])
        y_pred = reader.read()

        reader = ImageReader(path=gt_path.format(name, name),
                            image_type=itk.Image[itk.UC, 3])
        y_true = reader.read()

        df = main(name, y_pred, y_true)

        if os.path.exists(df_filename):
                dp = pd.read_csv(df_filename, sep=',', index_col=False)
                dp = dp.append(df, ignore_index=True)
                dp.to_csv(df_filename, sep=',', index=False)
        else:
            df.to_csv(df_filename, sep=',', index=False)
