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
from FemurSegmentation.image_splitter import LegImages



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


def main(image) :

    # pre_process the image
    #
    pass


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

        leg = get_labeled_leg(leg1, leg2)

        l, _ = image2array(leg[0])
        m, _ = image2array(leg[1])

        print('Shape: {}, {}'.format(l.shape, m.shape), flush = True)
        print('Values: {}'.format(np.unique(m)), flush = True)
