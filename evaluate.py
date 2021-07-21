#!/bin/env python

import os
import itk
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

from FemurSegmentation.utils import image2array, array2image, cast_image
from FemurSegmentation.IOManager import ImageReader
from FemurSegmentation.filters import execute_pipeline
from FemurSegmentation.filters import label_image2shape_label_map
from FemurSegmentation.metrics import dice_score

# %%

df_filename = r'D:\FemurSegmentation\DATA\graph_cut_results\not_optimized.csv'
mask_path = r'D:\FemurSegmentation\DATA\graph_cut_results\not_optimized\{}.nrrd'
prf_path = r'D:\FemurSegmentation\DATA\graph_cut_results\Profiles/{}_{}.npy'
gt_path = r'D:\FemurSegmentation\DATA\Input\{}\{}_gt.nrrd'


def view(image, idx=0, figsize=(10, 10)):
    '''
    '''
    if isinstance(image, np.ndarray):
        array = image.copy()
    else:
        array = itk.GetArrayFromImage(image)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    _ = ax.axis('off')
    _ = ax.imshow(array[idx], cmap='gray')

# %%


def save_array(filename, array):

    with open(filename, 'wb') as fp:
        np.save(fp, array)
# %%


def get_names(path, split_condition='.'):
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

# %%


def region_of_interest(image, region):

    PixelType, Dim = itk.template(image)[1]
    ImageType = itk.Image[PixelType, Dim]

    roi = itk.RegionOfInterestImageFilter[ImageType, ImageType].New()
    _ = roi.SetInput(image)
    _ = roi.SetRegionOfInterest(region)

    return roi

# %%


def adjust_physical_space(in_image, ref_image, ImageType):
    '''
    '''

    NNInterpolatorType = itk.NearestNeighborInterpolateImageFunction[ImageType,
                                                                     itk.D]
    interpolator = NNInterpolatorType.New()

    TransformType = itk.IdentityTransform[itk.D, 3]
    transformer = TransformType.New()
    _ = transformer.SetIdentity()

    resampler = itk.ResampleImageFilter[ImageType, ImageType].New()
    _ = resampler.SetInterpolator(interpolator)
    _ = resampler.SetTransform(transformer)
    _ = resampler.SetUseReferenceImage(True)
    _ = resampler.SetReferenceImage(ref_image)
    _ = resampler.SetInput(in_image)
    _ = resampler.Update()

    return resampler.GetOutput()

# %%


def get_hip_joint_region(ref_image, in_image):
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
    im = cast_image(ref_image, itk.UC)
    lmap = label_image2shape_label_map(im)
    lmap = execute_pipeline(lmap)
    bbox = lmap.GetNthLabelObject(0).GetBoundingBox()

    start = bbox.GetIndex()
    end = bbox.GetUpperIndex() - 1

    start = [start[0], start[1], (2 * end[2]) // 3]

    RegionType = itk.ImageRegion[3]

    region = RegionType()

    region.SetIndex(start)
    region.SetUpperIndex(end)

    ref_hip = region_of_interest(ref_image, region)
    seg_hip = region_of_interest(in_image, region)

    ref_hip = execute_pipeline(ref_hip)
    seg_hip = execute_pipeline(seg_hip)

    return ref_hip, seg_hip

# %%

def hausdorff_distance(ref_image, in_image, ImageType):

    hd = itk.HausdorffDistanceImageFilter[ImageType, ImageType].New()
    _ = hd.SetUseImageSpacing(True)
    _ = hd.SetInput1(ref_image)
    _ = hd.SetInput2(in_image)
    _ = hd.Update()

    return hd

# %%


def hd_slice_by_slice(ref_image, in_image, ImageType=itk.Image[itk.UC, 3]):

    lpr = ref_image.GetLargestPossibleRegion()
    max_index = lpr.GetSize()[2]
    upper_index = lpr.GetUpperIndex()
    RegionType = itk.ImageRegion[3]

    hDistance = []
    AVHD = []

    for i in range(0, max_index):

        try:

            region = RegionType()
            _ = region.SetIndex([0, 0, i])
            _ = region.SetUpperIndex([upper_index[0], upper_index[1], i])

            ref_slice = execute_pipeline(region_of_interest(ref_image, region))
            seg_slice = execute_pipeline(region_of_interest(in_image, region))

            hd = hausdorff_distance(ref_slice, seg_slice, ImageType)

            hDistance.append(hd.GetHausdorffDistance())
            AVHD.append(hd.GetAverageHausdorffDistance())
        except:
            hDistance.append(np.nan)
            AVHD.append(np.nan)

            continue

    return hDistance, AVHD


# %%


def evaluate_slice_by_slice(image1, image2):
    '''
    '''
    im1, _ = image2array(image1)
    im2, _ = image2array(image2)
    scores = [dice_score(i1, i2) for i1, i2 in zip(im1, im2)]

    return scores

# %%


def evaluate(image1, image2):

    ImageType = itk.Image[itk.SS, 3]
    # first make images in the same physical space
    adj = adjust_physical_space(ref_image=image1,
                                in_image=image2,
                                ImageType=ImageType)
    # now get the hip-joint region
    ref_hip, seg_hip = get_hip_joint_region(image1, adj)


    # now compute the metrics:
        # overall

    overall_dice = dice_score(image1, image2)
    overall_hd = hausdorff_distance(ref_image=image1,
                                    in_image=adj,
                                    ImageType=ImageType)

    # now only on the hip region
    hip_dice = dice_score(ref_hip, seg_hip)
    hip_hd = hausdorff_distance(ref_image=ref_hip,
                                in_image=seg_hip,
                                ImageType=ImageType)
    # profile
    dice_profile = evaluate_slice_by_slice(image1, adj)
    hd_profile, ahd_profile = hd_slice_by_slice(ref_image=image1,
                                                in_image=adj,
                                                ImageType=ImageType)

    res = {'overall dice' : [overall_dice],
            'overall hd' : [overall_hd.GetHausdorffDistance()],
            'overall avg hd' : [overall_hd.GetAverageHausdorffDistance()],
            'hip dice' : [hip_dice],
            'hip hd' : [hip_hd.GetHausdorffDistance()],
            'hip avg hd' : [hip_hd.GetAverageHausdorffDistance()]}

    return res, hd_profile, ahd_profile


def main(name, y_pred, y_true):

    res, hd_profile, ahd_profile = evaluate(y_true, y_pred)

    df = pd.DataFrame(res)
    df['Patient'] = name

    return df, hd_profile, ahd_profile


if __name__ == '__main__':

    names = get_names(path=mask_path)

    for name in names:
        print('I am processing {}'.format(name), flush=True)

        reader = ImageReader(path=mask_path.format(name),
                            image_type=itk.Image[itk.SS, 3])
        y_pred = reader.read()

        reader = ImageReader(path=gt_path.format(name, name),
                            image_type=itk.Image[itk.SS, 3])
        y_true = reader.read()

        try:
            df, hd_profile, ahd_profile = main(name, y_pred, y_true)

            _ = save_array(prf_path.format(name, 'hd'), hd_profile)
            _ = save_array(prf_path.format(name, 'avhd'), ahd_profile)

            if os.path.exists(df_filename):
                dp = pd.read_csv(df_filename, sep=',', index_col=False)
                dp = dp.append(df, ignore_index=True)
                dp.to_csv(df_filename, sep=',', index=False)
            else:
                df.to_csv(df_filename, sep=',', index=False)
        except:
            print('{} has excepted'.format(name))
