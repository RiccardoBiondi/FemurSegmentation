import itk
import numpy as np
from FemurSegmentation.utils import get_image_spatial_info, set_image_spatial_info
from FemurSegmentation.utils import image2array
from FemurSegmentation.filters import execute_pipeline
import matplotlib.pyplot as plt


__author__ = ['Riccardo Biondi']
__email__ = ['riccardo.biondi7@unibo.it']


def view(image, idx=0):
    array, _ = image2array(image)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    _ = ax.axis('off')
    _ = ax.imshow(array[idx], cmap='gray')


def dice_score(image1, image2):
    '''
    '''
    if isinstance(image1, np.ndarray):
        y_true = image1
    else :
        y_true = itk.GetArrayFromImage(image1)

    if isinstance(image2, np.ndarray):
        y_pred = image2
    else:
        y_pred = itk.GetArrayFromImage(image2)

    den = np.sum(y_true) + np.sum(y_pred)

    intersection = np.logical_and(y_true, y_pred)

    num = 2 * np.sum(intersection)
    return num / den


def danielsson_distance_map(image, square_distance=False, use_image_spacing=False):
    '''
    '''
    array, _ = image2array(image)
    PixelType, Dim = itk.template(image)[1]
    InputType = itk.Image[PixelType, Dim]
    OutputType = itk.Image[itk.F, Dim]

    dm = itk.DanielssonDistanceMapImageFilter[InputType, OutputType].New()
    _ = dm.SetInput(image)
    _ = dm.SetSquaredDistance(square_distance)
    _ = dm.SetUseImageSpacing(use_image_spacing)
    if len(np.unique(array)) == 2:
        _ = dm.InputIsBinaryOn()

    return dm


def housdorff_distance(image1, image2, mode='HD', **kwargs) :
    '''
    Compute the housdorff distance between two binary images
    Note the two images MUST have the same shape

    Parameters
    ----------
    image1: itk.Image
        first image
    image2: itk.Image
        second image
    mode:
        string which select the kind of distance to compute:

        'average' : compute the average housdorff distance
        'HD' : compute the classical housdorff distance
        'quantile' : compute the quantile housdorff distance with q=0.95

    **kwargs: kwargs for danielsson_distance_map function
    '''
    # make shure that the specified mode is supported
    if mode not in ['average', 'HD', 'quantile']:
        raise ValueError('Mode {} not supported'.format(mode))
    # now start to compute
    dX = danielsson_distance_map(image1, **kwargs)
    dY = danielsson_distance_map(image2, **kwargs)

    dX = execute_pipeline(dX)
    dY = execute_pipeline(dY)

    # get array from images
    X, _ = image2array(image1)
    Y, _ = image2array(image2)
    dX, _ = image2array(dX)
    dY, _ = image2array(dY)

    # depending to the specified parameter:
    if mode == 'HD':
        hXY = np.max(dX[Y==1])
        hYX = np.max(dY[X==1])
    elif mode =='average':
        hXY = np.mean(dX[Y==1])
        hYX = np.mean(dY[X==1])
    elif mode == 'quantile':
        hXY = np.quantile(dX[Y==1], 0.95)
        hYX = np.quantile(dY[X==1], 0.95)

    return np.max([hXY, hYX])


if __name__ == '__main__':

    from FemurSegmentation.IOManager import ImageReader
    gt_path = '/mnt/d/Riccardo/Progetti/Aperti/FemurSegmentation/Data/Formatted/D0062/D0062_gt.nrrd'
    gc_path = '/mnt/d/Riccardo/Progetti/Aperti/FemurSegmentation/Data/Formatted/D0062/D0062_gc.nrrd'

    reader = ImageReader(gt_path, itk.Image[itk.UC, 3])
    gt = reader.read()

    reader = ImageReader(gc_path, itk.Image[itk.UC, 3])
    gc = reader.read()

    hd = housdorff_distance(gc, gt, use_image_spacing=True)

    print(hd)
