#!/usr/bin/env python
# -*- coding: utf-8 -*

# test library
import pytest
import hypothesis.strategies as st
from hypothesis import given, settings
from  hypothesis import HealthCheck as HC

#useful libraries
import os
import itk
import numpy as np

# third part
from FemurSegmentation.utils import image2array, get_image_spatial_info
from FemurSegmentation.filters import gaussian_smoothing
from FemurSegmentation.filters import hessian_matrix
from FemurSegmentation.filters import get_eigenvalues_map
from FemurSegmentation.filters import execute_pipeline
# to test
from FemurSegmentation.boneness import Boneness



__author__ = ['Riccardo Biondi']
__email__  = ['riccardo.biondi4@studio.unibo.it']



# ███████ ████████ ██████   █████  ████████ ███████  ██████  ██ ███████ ███████
# ██         ██    ██   ██ ██   ██    ██    ██      ██       ██ ██      ██
# ███████    ██    ██████  ███████    ██    █████   ██   ███ ██ █████   ███████
#      ██    ██    ██   ██ ██   ██    ██    ██      ██    ██ ██ ██           ██
# ███████    ██    ██   ██ ██   ██    ██    ███████  ██████  ██ ███████ ███████

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
def roi_strategy(draw) :

    image = np.random.rand(200, 200, 200)
    image = (image < .5).astype(np.uint8)
    image = itk.GetImageFromArray(image)
    return image


@st.composite
def scales_strategy(draw) :
    len_ = draw(st.integers(1, 5))
    scales = []

    for i in range(len_) :
        scales.append(draw(st.floats(0.3, 1.7)))
    return scales

    # ████████ ███████ ███████ ████████
    #    ██    ██      ██         ██
    #    ██    █████   ███████    ██
    #    ██    ██           ██    ██
    #    ██    ███████ ███████    ██


class TestBoneness :


    @given(random_image_strategy(), scales_strategy())
    @settings(max_examples = 20, deadline = None,
              suppress_health_check = (HC.too_slow, ))
    def test_Init_woRoi(self, image, scales) :
        '''
        Given:
            - itk Image
            - list of scales
        Then:
            - initialize a Boneness obj
        Assert:
            - image correctly initialized
            - scales correctly initialized
            - roi == None
        '''
        inImage, inInfo = image2array(image)
        bones = Boneness(image, scales)
        setImage, setInfo = image2array(bones.image)

        assert np.all(np.isclose(inImage, setImage))
        assert inInfo == setInfo
        assert np.all(scales == bones.scales)



    @given(random_image_strategy(), scales_strategy(), roi_strategy())
    @settings(max_examples = 20, deadline = None,
              suppress_health_check = (HC.too_slow, ))
    def test_Init_wRoi(self, image, scales, roi) :
        '''
        Given:
            - itk Image
            - scales list
            - roi image
        Then:
            - create a Boneness object
        Assert:
            - image is correctly setted
            - scales are correctly setted
            - roi is correctly setted and of type np.array
        '''
        inImage, inInfo = image2array(image)
        inRoi, _ = image2array(roi)
        bones = Boneness(image, scales, roi)
        setImage, setInfo = image2array(bones.image)

        assert np.all(np.isclose(inImage, setImage))
        assert inInfo == setInfo
        assert np.all(scales == bones.scales)
        #assert isinstance(bones.roi, type(np.array)) # FIXME
        assert np.all(bones.roi == inRoi)

    @given(random_image_strategy(), st.floats(0.1, 1.5))
    @settings(max_examples = 20, deadline = None,
              suppress_health_check = (HC.too_slow, ))
    def test_eigenvaluesMeasures(self, image, scale) :
        '''
        Given:
            - itk image
            - scale
        Then:
            - compute the image eigenvalues map
            - compute the eigenvalues measures: R_bones, R_noise, eigen_no_null
        Assert:
            - correct eigen no null measure
            - R_bones in range []
            - R_noise in range []
        '''
        # in this case image and scale act as a place holder, since alle the
        # required quantities were computed using external methods
        # the method to test, defined in Boneness, requires the eigenmap
        bones = Boneness(image, scale)

        sm = gaussian_smoothing(image, sigma = scale)
        hess = hessian_matrix(sm.GetOutput(), sigma = scale)
        pipe = get_eigenvalues_map(hess.GetOutput())
        eigen_map = execute_pipeline(pipe)

        R_bones, R_noise, eigen_no_null, eigen = bones.computeEigenvaluesMeasures(eigen_map)
        eg_abs = np.abs(eigen)

        assert ~np.isclose(eigen[eigen_no_null, 2], 0)
        assert np.all(R_noise[eigen_no_null] > eg_abs[eigen_no_null, 2])
        assert np.all(R_noise[eigen_no_null] < 3 * eg_abs[eigen_no_null, 2])

        assert np.all(R_bones[eigen_no_null] > 0)
        assert np.all(R_bones[eigen_no_null] < 1)


    # TODO add more tests (also for the ohter methods!)
