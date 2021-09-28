#!/bin/env python

import os
import itk
import argparse
from glob import glob

import matplotlib.pyplot as plt

from FemurSegmentation.IOManager import ImageReader, VolumeWriter
from FemurSegmentation.image_splitter import LegImages


__author__ = ['Riccardo Biondi']
__email__ = ['riccardo.biondi7@unibo.it']


def parse_args():
    description = 'Prepare data for segmentation'

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--input',
                        dest='input',
                        required=True,
                        type=str,
                        action='store',
                        help='Input filename')
    parser.add_argument('--output',
                        dest='output',
                        type=str,
                        action='store',
                        required=True,
                        help='output filename')
    parser.add_argument('--label',
                        dest='label',
                        type=str,
                        action='store',
                        required=False,
                        default=None,
                        help='Possible labeled image')

    args = parser.parse_args()

    return args


def main():

    args = parse_args()


    print('I am reading from {}'.format(args.input), flush=True)

    reader = ImageReader()

    image = reader(args.input, itk.Image[itk.SS, 3])

    if args.label is not None:
        print('I am reading the label from: {}'.format(args.label), flush=True)

        label = reader(args.label, itk.Image[itk.UC, 3])

    else:
        label = None

    print('I am splitting', flush=True)

    splitter = LegImages(image=image, mask=label)
    leg1, leg2 = splitter.get_legs()

    out_name = os.path.basename(args.input)
    print(out_name, flush=True)

    mask_1 = '_'.join([args.output, 'Label1.nrrd'])
    mask_2 = '_'.join([args.output, 'Label2.nrrd'])

    image_1 = '_'.join([args.output, '1.nrrd'])
    image_2 = '_'.join([args.output, '2.nrrd'])


    writer = VolumeWriter()

    if isinstance(leg1, tuple) and isinstance(leg2, tuple):
        print('I am saving..', flush=True)

        print('Saving Right Leg to: {}'.format(image_1), flush=True)
        _ = writer(image_1, leg1[0])
        print('Saving Left Mask to: {}'.format(mask_1), flush=True)
        _ = writer(mask_1, leg1[1])

        print('Saving Right Leg to: {}'.format(image_2), flush=True)
        _ = writer(image_2, leg2[0])
        print('Saving Left Mask to: {}'.format(mask_2), flush=True)
        _ = writer(mask_2, leg2[1])

    else:
        print('Saving Right Leg to: {}'.format(image_1), flush=True)
        _ = writer(image_1, leg1)

        print('Saving Left Leg to: {}'.format(image_2), flush=True)
        _ = writer(image_2, leg2)

    print('[DONE]', flush=True)







if __name__ == '__main__' :
    main()
