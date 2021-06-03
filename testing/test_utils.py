#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import hypothesis

import pytest
import hypothesis.strategies as st
from hypothesis import given, settings
from  hypothesis import HealthCheck as HC

import itk
import numpy as np

# import function to test
from FemurSegmentation.utils import cast_image
from FemurSegmentation.utils import image2array
from FemurSegmentation.utils import array2image
from FemurSegmentation.utils import get_image_spatial_info
from FemurSegmentation.utils import set_image_spatial_info
from FemurSegmentation.utils import get_labeled_leg


__author__ = ['Biondi Riccardo']
__email__ = ['riccardo.biondi7@unibo.it']

################################################################################
###                                                                          ###
###                         Define Test strategies                           ###
###                                                                          ###
################################################################################
pixel_types = [itk.UC, itk.SS, itk.F, itk.D]
image_dimensions = [2, 3]


@st.composite
def itk_image_strategy(draw) :
    '''
    Create a random image (or volume) with random image type
    '''

    PixelType = draw(st.sampled_from(pixel_types))
    Dimension = draw(st.sampled_from(image_dimensions))

    PixelType = draw(st.sampled_from(pixel_types))
    Dimension = draw(st.sampled_from(image_dimensions))
    ImageType = itk.Image[PixelType, Dimension]

    rndImage = itk.RandomImageSource[ImageType].New()
    rndImage.SetSize(200)
    rndImage.Update()

    return rndImage.GetOutput()


@st.composite
def itk_info_strategy(draw) :
    '''
    Ceate a dictionary with itk spatial information:
        - Direction
        - Origin
        - Spacing
        - Size
        - Index
        - Upper Index
    '''
    pass



@st.composite
def label_leg2_strategy(draw) :

    shape1 = (200, 200, 200)
    shape2 = (100, 100, 100)
    # leg one
    leg1 = np.random.rand(*shape1)
    lab1 = np.zeros(shape1)

    leg1 = itk.GetImageFromArray(leg1)
    lab1 = itk.GetImageFromArray(lab1)
    # leg two
    leg2 = np.random.rand(*shape2)
    lab2 = np.zeros(shape2)
    lab2[25 : 75, 25 : 75, 25 : 75] = np.ones((50, 50, 50))

    leg2 = itk.GetImageFromArray(leg2)
    lab2 = itk.GetImageFromArray(lab2)

    return (leg1, lab1), (leg2, lab2)



@st.composite
def label_leg1_strategy(draw) :
    shape1 = (200, 200, 200)
    shape2 = (100, 100, 100)
    # leg one
    leg1 = np.random.rand(*shape1)
    lab1 = np.zeros(shape1)
    lab1[25 : 75, 25 : 75, 25 : 75] = np.ones((50, 50, 50))
    leg1 = itk.GetImageFromArray(leg1)
    lab1 = itk.GetImageFromArray(lab1)

    # leg two
    leg2 = np.random.rand(*shape2)
    lab2 = np.zeros(shape2)

    leg2 = itk.GetImageFromArray(leg2)
    lab2 = itk.GetImageFromArray(lab2)


    return (leg1, lab1), (leg2, lab2)

################################################################################
###                                                                          ###
###                                 TESTING                                  ###
###                                                                          ###
################################################################################



@given(itk_image_strategy(), st.sampled_from(pixel_types))
@settings(max_examples = 20, deadline = None,
          suppress_health_check = (HC.too_slow, ))
def test_castImage(image, new_pixel_type) :
    '''
    Given :
        - itk image
        - new pixel type
    Then :
        - apply cast image
    Assert :
        - pixel type is changed correctly
        - image dimension is preserved
    '''
    newImage = cast_image(image, new_pixel_type)
    newImagePixelType, newImageDimension = itk.template(newImage)[1]
    _, oldImageDimension = itk.template(image)[1]

    assert newImagePixelType == new_pixel_type
    assert newImageDimension == oldImageDimension


#@given()
#@settings()
#def test_Image2Array() :
#    pass


#@given()
#@settings()
#def test_Array2Image() :
#    pass




@given(label_leg2_strategy())
def test_select_leg2_w_wlabel(legs) :
    '''
    Given:
        - tuple with labels
        - leg2 corresponding to labeled one
    Then :
        - select only the one labeled
    Assert :
        - correct selection is made
    '''
    leg1 = legs[0]
    leg2 = legs[1]

    selected = get_labeled_leg(leg1, leg2)

    assert np.all(selected[0] == leg2[0])



@given(label_leg1_strategy())
def test_select_leg1_w_wlabel(legs) :
    '''
    Given:
        - tuple with labels
        - leg1 corresponding to labeled one
    Then :
        - select only the one labeled
    Assert :
        - correct selection is made
    '''
    leg1 = legs[0]
    leg2 = legs[1]

    selected = get_labeled_leg(leg1, leg2)

    assert np.all(selected[0] == leg1[0])
