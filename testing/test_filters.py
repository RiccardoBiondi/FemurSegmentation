#!/usr/bin/env python
# -*- coding: utf-8 -*

# test base
import pytest
import hypothesis.strategies as st
from hypothesis import given, settings
from  hypothesis import HealthCheck as HC

# useful
import os
import itk
import numpy as np

# third part
from FemurSegmentation.utils import image2array, array2image

#finction to test
from FemurSegmentation.filters import binary_threshold
from FemurSegmentation.filters import threshold
from FemurSegmentation.filters import median_filter
from FemurSegmentation.filters import gaussian_smoothing
from FemurSegmentation.filters import hessian_matrix
from FemurSegmentation.filters import get_eigenvalues_map
from FemurSegmentation.filters import connected_components #TODO test me
from FemurSegmentation.filters import relabel_components # TODO test me



__author__ = ['Riccardo Biondi']
__email__  = ['riccardo.biondi7@unibo.it']


# ███████ ████████ ██████   █████  ████████ ███████  ██████  ██ ███████ ███████
# ██         ██    ██   ██ ██   ██    ██    ██      ██       ██ ██      ██
# ███████    ██    ██████  ███████    ██    █████   ██   ███ ██ █████   ███████
#      ██    ██    ██   ██ ██   ██    ██    ██      ██    ██ ██ ██           ██
# ███████    ██    ██   ██ ██   ██    ██    ███████  ██████  ██ ███████ ███████

pixel_types = [itk.UC, itk.SS]
out_types = [itk.UC, itk.SS, itk.F]

@st.composite
def random_image_strategy(draw) :
    PixelType = draw(st.sampled_from(pixel_types))
    ImageType = itk.Image[PixelType, 3]

    rndImage = itk.RandomImageSource[ImageType].New()
    rndImage.SetSize(200)
    rndImage.Update()

    return rndImage.GetOutput()



    # ████████ ███████ ███████ ████████
    #    ██    ██      ██         ██
    #    ██    █████   ███████    ██
    #    ██    ██           ██    ██
    #    ██    ███████ ███████    ██



@given(random_image_strategy(), st.integers(20, 30), st.integers(200, 250))
@settings(max_examples = 20, deadline = None,
          suppress_health_check = (HC.too_slow, ))
def test_binary_threshold_upper_lower(image, lower_thr, upper_thr) :
    '''
    Given :
        - itk image obj
        - lower threshold value
        - upper threshold value
    Then :
        - apply binary threshold
    Assert :
        - binary image as output
        - masked original image value between upper and lower thr
    '''
    thr = binary_threshold(image, upper_thr, lower_thr)
    thr = itk.GetArrayFromImage(thr)
    to_mask = itk.GetArrayFromImage(image)

    assert np.all(np.unique(thr) == [0, 1])
    assert np.max(to_mask[thr != 0]) < upper_thr
    assert np.min(to_mask[thr != 0]) > lower_thr


@given(random_image_strategy(), st.integers(20, 30), st.integers(200, 250))
@settings(max_examples = 20, deadline = None,
          suppress_health_check = (HC.too_slow, ))
def test_binary_threshold_inside_outside(image, inside, outside) :
    '''
    Given:
        - itk image obj
        - inside value
        - outside value
    Then:
        - apply binary threshold
    Assert:
        - binary image values are inside and outside
    '''
    thr = binary_threshold(image, 230, 15, inside, outside)
    thr = itk.GetArrayFromImage(thr)

    assert np.all(np.unique(thr) == [inside, outside])



@given(random_image_strategy(), st.sampled_from(out_types))
@settings(max_examples = 20, deadline = None,
          suppress_health_check = (HC.too_slow, ))
def test_binary_threshold_output_type(image, out_type) :
    '''
    Given :
        - itk image obj
        - itk pixel type
    Then :
        - apply threshold specyfiyng the output type
    Assert:
        - output type match the specified one
    '''
    thr = binary_threshold(image, 20, 150, out_type = out_type)
    PixelType, _ = itk.template(thr)[1]

    assert PixelType == out_type


@given(random_image_strategy(), st.integers(1, 5))
@settings(max_examples = 20, deadline = None,
          suppress_health_check = (HC.too_slow, ))
def test_median_filter(image, radius) :
    '''
    Given :
        - itk Image
        - radius
    Then :
        - call median filter function
    Assert :
        - image is correctly setted
        - redius is correctly setted
    '''
    median = median_filter(image, radius)
    inArray, inInfo = image2array(image)

    setArray, setInfo = image2array(median.GetInput())

    assert np.all(median.GetRadius() == [radius] * 3)
    assert np.all(np.isclose(setArray, inArray))
    assert inInfo == setInfo



@settings(max_examples = 20, deadline = None,
          suppress_health_check = (HC.too_slow, ))
@given(random_image_strategy(), st.floats(0.1, 1.5), st.booleans())
def test_gaussian_smoothing(image, sigma, normalize_across_scale) :
    '''
    Given:
        - image
        - sigma
        - flag
    Then :
        - apply gaussian_smoothing
    Assert :
        - image is correctly setted
        - sigma is correctly setted
        - flag is correctly setted
    '''
    smooth = gaussian_smoothing(image, sigma, normalize_across_scale)

    inArr, inInfo = image2array(image)
    setArr, setInfo = image2array(smooth.GetInput())

    assert np.all(np.isclose(inArr, setArr))
    assert inInfo == setInfo
    assert smooth.GetSigma() == sigma
    assert smooth.GetNormalizeAcrossScale() == normalize_across_scale



@settings(max_examples = 20, deadline = None,
          suppress_health_check = (HC.too_slow, ))
@given(random_image_strategy(), st.floats(0.1, 1.5), st.booleans())
def test_hessian_matrix(image, sigma, normalize_across_scale) :
    '''
    Given:
        - image
        - sigma
        - flag
    Then :
        - compute hessian
    Assert :
        - image is correctly setted
        - sigma is correctly setted
        - flag is correctly setted
    '''
    hessian = hessian_matrix(image, sigma, normalize_across_scale)

    inArr, inInfo = image2array(image)
    setArr, setInfo = image2array(hessian.GetInput())

    assert np.all(np.isclose(inArr, setArr))
    assert inInfo == setInfo
    assert hessian.GetSigma() == sigma
    assert hessian.GetNormalizeAcrossScale() == normalize_across_scale



@settings(max_examples = 20, deadline = None,
          suppress_health_check = (HC.too_slow, ))
@given(random_image_strategy(), st.floats(0.1, 0.5), st.integers(1, 3))
def test_eigenvalues(image, sigma, order ) :
    '''
    Given :
        - itk.Image
        - sigma
        - order
    Then :
        - compute the hessian matrix
        - initialize the eigenvalues filter
    Assert :
        - the filter parameters are correctly initialized
    '''
    #create, update and get the output of the hessian filter
    # that because we need to provide an hessian matrix as input to the
    # testing filter
    hess_filter = hessian_matrix(image, sigma)
    _ = hess_filter.Update()
    hess = hess_filter.GetOutput()

    arr1, info1 = image2array(hess)

    eigen = get_eigenvalues_map(hess, order = order)

    arr2, info2 = image2array(eigen.GetInput())

    # in this assertion I am using the function np.isclose because the two array
    #to match are of floating point type, so there can be some imprecisions due
    # computer eps
    assert np.all(np.isclose(arr1, arr2))
    assert info1 == info2
