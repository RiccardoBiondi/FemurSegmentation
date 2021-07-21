#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import hypothesis.strategies as st
from hypothesis import given, settings
from hypothesis import HealthCheck as HC

import os
import itk
import numpy as np
import FemurSegmentation.IOManager as IOManager


__author__ = ['Riccardo Biondi']
__email__ = ['riccardo.biondi4@studio.unibo.it']


# ███████ ████████ ██████   █████  ████████ ███████  ██████  ██ ███████ ███████
# ██         ██    ██   ██ ██   ██    ██    ██      ██       ██ ██      ██
# ███████    ██    ██████  ███████    ██    █████   ██   ███ ██ █████   ███████
#      ██    ██    ██   ██ ██   ██    ██    ██      ██    ██ ██ ██           ██
# ███████    ██    ██   ██ ██   ██    ██    ███████  ██████  ██ ███████ ███████


legitimate_chars = st.characters(whitelist_categories=('Lu', 'Ll'),
                                min_codepoint=65, max_codepoint=90)

text_strategy = st.text(alphabet=legitimate_chars, min_size=1,
                        max_size=15)

medical_image_format = ['nrrd', 'nii']
pixel_types = [itk.UC, itk.SS]


@st.composite
def random_image_strategy(draw) :
    PixelType = draw(st.sampled_from(pixel_types))
    ImageType = itk.Image[PixelType, 3]

    rndImage = itk.RandomImageSource[ImageType].New()
    rndImage.SetSize(200)
    rndImage.Update()

    return rndImage.GetOutput()


@st.composite
def path2image_strategy(draw) :
    folder_name = draw(text_strategy)
    image_name = draw(text_strategy)
    image_format = draw(st.sampled_from(medical_image_format))

    return folder_name, image_name, image_format

    # ████████ ███████ ███████ ████████
    #    ██    ██      ██         ██
    #    ██    █████   ███████    ██
    #    ██    ██           ██    ██
    #    ██    ███████ ███████    ██


# ██████  ███████  █████  ██████  ███████ ██████
# ██   ██ ██      ██   ██ ██   ██ ██      ██   ██
# ██████  █████   ███████ ██   ██ █████   ██████
# ██   ██ ██      ██   ██ ██   ██ ██      ██   ██
# ██   ██ ███████ ██   ██ ██████  ███████ ██   ██

class TestImageReader :
    '''
    '''
    def test_ImageReaderDefaultInit(self) :
        '''
        Check if the ImageReader object is correctly initialized with the
        default parameters
        '''
        reader = IOManager.ImageReader()

        assert reader.path == ""
        assert reader.image_type == itk.Image[itk.UC, 3]

    @given(path2image_strategy(), st.sampled_from(pixel_types))
    @settings(max_examples=20, deadline=None,
              suppress_health_check=(HC.too_slow, ))
    def test_ImageReaderParamsInit(self, imagePath, pixelType):
        '''
        Given :
            - path
            - PixelType
            - Image Dimension
        Then :
            - Set reader path to path
            - Set image type to image type
        Assert :
            - Correct assignement
        '''

        base_path = "./testing/test_images/{}/{}.{}"
        path = base_path.format(imagePath[0], imagePath[1], imagePath[2])
        ImageType = itk.Image[pixelType, 3]
        reader = IOManager.ImageReader(path, ImageType)

        assert reader.path == path
        assert reader.image_type == ImageType

    def test_isPath2File(self) :
        '''
        Given a path to a folder, test that return False
        '''

        reader = IOManager.ImageReader("./testing/test_images",
                                    itk.Image[itk.SS, 3])
        assert ~reader.isPath2File()

    def test_isPath2File_IOError(self) :
        '''
        Given :
            - path to non existing file
        Assert :
            isPath2File raise IOError()
        '''
        reader = IOManager.ImageReader()
        with pytest.raises(OSError) :
            assert reader.isPath2File()

            # ██     ██ ██████  ██ ████████ ███████ ██████
            # ██     ██ ██   ██ ██    ██    ██      ██   ██
            # ██  █  ██ ██████  ██    ██    █████   ██████
            # ██ ███ ██ ██   ██ ██    ██    ██      ██   ██
            #  ███ ███  ██   ██ ██    ██    ███████ ██   ██


class TestVolumeWriter :

    def test_VolumeWriterDefaultInit(self) :
        '''
        Check if the VolumeWriter object is correctly initialized with the
        default parameters
        '''
        writer = IOManager.VolumeWriter()

        assert writer.path == ""
        assert writer.image is None
        assert ~writer.as_dicom

    @given(random_image_strategy(), path2image_strategy())
    def test_VolumeWriterParamsInit(self, image, inPath) :
        '''
        Given :
            - random image
            - path
            - output format
        Then :
            - Set writer path to path
            - Set writer image to image
            - Set writer out_format to output format
        Assert :
            - Correct assignement is done
        '''
        base_path = "./testing/test_images/{}/{}.{}"
        path = base_path.format(inPath[0], inPath[1], inPath[2])

        writer = IOManager.VolumeWriter(path, image)

        assert writer.path == path
        assert writer.image == image
        assert ~writer.as_dicom

        # ██████  ██     ██
        # ██   ██ ██     ██
        # ██████  ██  █  ██
        # ██   ██ ██ ███ ██
        # ██   ██  ███ ███


class TestReadAndWrite :

    @given(random_image_strategy(), path2image_strategy())
    @settings(max_examples=20, deadline=None,
              suppress_health_check=(HC.too_slow, ))
    def test_read_and_write_image(self, image, inPath) :
        '''
        Given :
            - itk Image
            - output filename
        Then :
            - write the image volume
            - read the image volume
        Assert :
            - the red image is equal to the input one
        '''
        base_path = "./testing/test_images/{}.{}"
        path = base_path.format(inPath[1], inPath[2])
        pixelType, dimension = itk.template(image)[1]
        imageType = itk.Image[pixelType, dimension]
        inArray = itk.GetArrayFromImage(image)

        writer = IOManager.VolumeWriter(path=path, image=image)
        _ = writer.volume2Image()

        reader = IOManager.ImageReader(path=path, image_type=imageType)
        redImage = reader.image2Volume()

        redPixel, redDim = itk.template(redImage)[1]
        redArray = itk.GetArrayFromImage(redImage)

        os.remove(path) # remove the file generated for the test


        assert redPixel == pixelType
        assert redDim == dimension
        assert np.isclose(redArray, inArray).all()

    @given(random_image_strategy(), text_strategy)
    @settings(max_examples=20, deadline=None,
              suppress_health_check=(HC.too_slow, ))
    def test_read_and_write_dicom(self, image, folder_name) :
        '''
        Given :
            - itk 3D image
            - folder name
        Then :
            - write the image as dicom series
            - read the dicom series
        Assert :
            - red dimension are equal to input one
            - red pixel type is equal to input one
            - red image array is close to input one
        '''
        dicom_path = os.path.join("./testing/test_images", folder_name)
        pixelType, dimension = itk.template(image)[1]
        imageType = itk.Image[pixelType, dimension]
        inArray = itk.GetArrayFromImage(image)

        writer = IOManager.VolumeWriter(path=dicom_path, image=image,
                                        as_dicom=True)
        _ = writer.volume2DICOM()

        reader = IOManager.ImageReader(path=dicom_path, image_type=imageType)
        redImage = reader.DICOM2Volume()

        redPixel, redDim = itk.template(redImage)[1]
        redArray = itk.GetArrayFromImage(redImage)

        for f in os.listdir(dicom_path):
            os.remove(os.path.join(dicom_path, f))
        os.rmdir(dicom_path)


        assert redPixel == pixelType
        assert redDim == dimension
        assert np.isclose(redArray, inArray).all()


    @given(random_image_strategy(), path2image_strategy(), st.booleans())
    @settings(max_examples=20, deadline=None,
              suppress_health_check=(HC.too_slow, ))
    def test_read_and_write(self, image, inPath, as_dicom) :
        '''
        Given :
            - itk image
            - output path
            - boolean
        Then :
            - write the image
            - read the image
        Assert :
            - red image pixel type is equal to input one
            - red image dimension is equal to input one
            - red image array is close to the input one
        '''
        pixelType, dimension = itk.template(image)[1]
        imageType = itk.Image[pixelType, dimension]
        inArray = itk.GetArrayFromImage(image)

        if as_dicom :
            path = os.path.join("./testing/test_images", inPath[0])
        else :
            base_path = "./testing/test_images/{}.{}"
            path = base_path.format(inPath[1], inPath[2])

        writer = IOManager.VolumeWriter(path=path, image=image, as_dicom=as_dicom)
        _ = writer.write()
        reader = IOManager.ImageReader(path=path, image_type=imageType)
        redImage = reader.read()

        redPixelType, redDimension = itk.template(redImage)[1]
        redArray = itk.GetArrayFromImage(redImage)

        if as_dicom:
            for f in os.listdir(path):
                os.remove(os.path.join(path, f))
            os.rmdir(path)
        else:
            os.remove(path)

        assert redDimension == dimension
        assert redPixelType == pixelType
        assert np.isclose(inArray, redArray).all()

    @given(random_image_strategy(), path2image_strategy(), st.booleans())
    @settings(max_examples=20, deadline=None,
              suppress_health_check=(HC.too_slow, ))
    def test_read_and_write_call(self, image, inPath, as_dicom) :
        '''
        Given :
            - itk image
            - output path
            - boolean
        Then :
            - call writer as function and write the image
            - call reader as function and read the image
        Assert :
            - red image pixel type is equal to input one
            - red image dimension is equal to input one
            - red image array is close to the input one
        '''
        pixelType, dimension = itk.template(image)[1]
        imageType = itk.Image[pixelType, dimension]
        inArray = itk.GetArrayFromImage(image)

        if as_dicom :
            path = os.path.join("./testing/test_images", inPath[0])
        else :
            base_path = "./testing/test_images/{}.{}"
            path = base_path.format(inPath[1], inPath[2])

        writer = IOManager.VolumeWriter()
        _ = writer(path=path, image=image, as_dicom=as_dicom)
        reader = IOManager.ImageReader()
        redImage = reader(path=path, image_type=imageType)

        redPixelType, redDimension = itk.template(redImage)[1]
        redArray = itk.GetArrayFromImage(redImage)

        if as_dicom:
            for f in os.listdir(path):
                os.remove(os.path.join(path, f))
            os.rmdir(path)
        else:
            os.remove(path)

        assert redDimension == dimension
        assert redPixelType == pixelType
        assert np.isclose(inArray, redArray).all()
