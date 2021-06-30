import os
import itk
import argparse
from glob import glob

from FemurSegmentation.IOManager import ImageReader, VolumeWriter
from FemurSegmentation.image_splitter import LegImages
from FemurSegmentation.utils import get_labeled_leg

base = '/mnt/d/Riccardo/Progetti/Aperti/FemurSegmentation/Data/Scans/*'
image_base = '/mnt/d/Riccardo/Progetti/Aperti/FemurSegmentation/Data/Scans/{}/{}_CTData/'
mask_base = '/mnt/d/Riccardo/Progetti/Aperti/FemurSegmentation/Data/Scans/{}/{}_Segmentations/*.nii.gz'

out_fold = '/mnt/d/Riccardo/Progetti/Aperti/FemurSegmentation/Data/Input/{}/'
out_base = '/mnt/d/Riccardo/Progetti/Aperti/FemurSegmentation/Data/Input/{}/{}_{}.nrrd'


def get_names(path):

    paths = sorted(glob(path))
    names = list(map(lambda x : os.path.basename(x), paths))

    return names


def main():

    names = get_names(base)

    for name in names:

        print('I am processing... {}'.format(name), flush=True)
        m_path = glob(mask_base.format(name, name))[0]
        print(m_path)
        i_path = image_base.format(name, name)

        try:

            print("\tI am reading", flush=True)
            # read the image
            reader = ImageReader(path=i_path, image_type=itk.Image[itk.F, 3])
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

            writer = VolumeWriter( path=i_out, image=leg[0])
            _ = writer.write()

            writer = VolumeWriter(path=m_out, image=leg[1])
            _ = writer.write()

            print("[DONE]", flush=True)
        except:
            print('{} excepted'.format(name), flush=True)


if __name__ == '__main__':
    main()
