#!/bin/env python

import os
import itk
import argparse
from glob import glob

import matplotlib.pyplot as plt

from FemurSegmentation.IOManager import ImageReader, VolumeWriter
from FemurSegmentation.image_splitter import LegImages
from FemurSegmentation.utils import get_labeled_leg, cast_image


base = r'D:\FemurSegmentation\DATA\Scans\*'
image_base = r'D:\FemurSegmentation\DATA\Scans\{}\{}_CTData'
mask_base = r'D:\FemurSegmentation\DATA\Scans\{}\*{}'

out_fold = r'D:\FemurSegmentation\DATA\Input\{}'
out_base = r'D:\FemurSegmentation\DATA\Input\{}\{}_{}.nrrd'


def get_names(path):

    paths = sorted(glob(path))
    names = list(map(lambda x : os.path.basename(x), paths))

    return names


def view(image, idx=0):
    array = itk.GetArrayFromImage(image)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
    _ = ax.axis('off')
    _ = ax.imshow(array[idx], cmap='gray')


def main():

    names = get_names(base)


    for name in names:

        m_path = []
        print('I am processing... {}'.format(name), flush=True)
        try :
            m_path = glob(mask_base.format(name, '.nii.gz'))[0]
        except:
            m_path = glob(mask_base.format(name, '.nrrd'))[0]
        i_path = image_base.format(name, name)

        try:

            print("\tI am reading", flush=True)
            # read the image
            reader = ImageReader(path=i_path, image_type=itk.Image[itk.SS, 3])
            image = reader.read()

            # read the mask
            reader = ImageReader(path=m_path, image_type=itk.Image[itk.UC, 3])
            mask = reader.read()

            print("\tI am splitting", flush=True)
            # now split the iamge into leg
            splitter = LegImages(image=image, mask=mask)
            leg1, leg2 = splitter.get_legs()

            leg = get_labeled_leg(leg1, leg2)

            # write the image
            print("\tI am writing", flush=True)

            i_out = out_base.format(name, name, 'im')
            m_out = out_base.format(name, name, 'gt')



            try:
                os.mkdir(out_fold.format(name))
            except:
                continue

            # prchè non è in grado di scrivere i DICOM di tipo Float... Devo
            # fare ricerche
            im = cast_image(leg[0], itk.SS)

            writer = VolumeWriter(path=i_out, image=im)
            _ = writer.write()

            writer = VolumeWriter(path=m_out, image=leg[1])
            _ = writer.write()

            print("[DONE]", flush=True)
        except:
            print('{} excepted'.format(name), flush=True)


if __name__ == '__main__':
    main()
