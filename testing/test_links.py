#!/usr/bin/env python
# -*- coding: utf-8 -*

# test libraries
import hypothesis.strategies as st
from hypothesis import given, settings
from hypothesis import HealthCheck as HC

# useful libraries
import itk
import numpy as np

# third part
from FemurSegmentation.filters import binary_threshold
from FemurSegmentation.boneness import Boneness

# to test
from FemurSegmentation.links import GraphCutLinks


__author__ = ['Riccardo Biondi']
__email__ = ['riccardo.biondi7@unibo.it']


# ███████ ████████ ██████   █████  ████████ ███████  ██████  ██ ███████ ███████
# ██         ██    ██   ██ ██   ██    ██    ██      ██       ██ ██      ██
# ███████    ██    ██████  ███████    ██    █████   ██   ███ ██ █████   ███████
#      ██    ██    ██   ██ ██   ██    ██    ██      ██    ██ ██ ██           ██
# ███████    ██    ██   ██ ██   ██    ██    ███████  ██████  ██ ███████ ███████

pixel_types = [itk.UC, itk.SS]


@st.composite
def binary_image_strategy(draw):

    image = np.random.rand(200, 200, 200)
    image = (image < .5).astype(np.uint8)
    image = itk.GetImageFromArray(image)
    return image


@st.composite
def random_image_strategy(draw):
    PixelType = draw(st.sampled_from(pixel_types))
    ImageType = itk.Image[PixelType, 3]

    rndImage = itk.RandomImageSource[ImageType].New()
    _ = rndImage.SetSize(200)
    _ = rndImage.SetMin(0)
    _ = rndImage.SetMax(255)
    _ = rndImage.Update()

    return rndImage.GetOutput()


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


class TestGraphCutLinks:

    @given(random_image_strategy(), scales_strategy())
    @settings(max_examples=20, deadline=None,
              suppress_health_check=(HC.too_slow, ))
    def test_Init(self, image, scales) :
        '''
        Given :
            - itk image
            - scales
        Then:
            - compute the boneness
            - find roi
            - find obj
            - find bkg
            - init GraphCutLinks object
        Assert:
            - GraphCutLinks attribute correctly initialized
        '''
        # compute the required arguments for link initialization
        roi = binary_threshold(image, 200, 50)
        obj = binary_threshold(image, 200, 150)
        bkg = binary_threshold(image, 100, 50)

        bones = Boneness(image, scales, roi)
        boneness = bones.computeBonenessMeasure()

        inImage = itk.GetArrayFromImage(image)
        inRoi = itk.GetArrayFromImage(roi)

        inObj = itk.GetArrayFromImage(obj)
        inBkg = itk.GetArrayFromImage(bkg)
        inBoneness = itk.GetArrayFromImage(boneness)

        # initialize link
        link = GraphCutLinks(image, boneness, roi, obj, bkg)

        # in this case I am using np.isclose since the data are o float type
        assert np.all(np.isclose(inImage, link.image))
        assert np.all(np.isclose(inBoneness, link.boneness))
        # now data are of integer type, so I am not using the np.isclose
        # function
        assert np.all(inObj == link.obj)
        assert np.all(inBkg == link.bkg)
        assert np.all(inRoi == link.roi)
        # now check that the default initialization is good
        assert link.Lambda == 100.
        assert link.sigma == .25
        # check number of voxels and vx_id

    @given(random_image_strategy(), st.floats(0.1, 1.5))
    @settings(max_examples=20, deadline=None,
              suppress_health_check=(HC.too_slow, ))
    def test_tLinkSource(self, image, scale) :
        '''
        Given:
            - image
            - scale (single is enough)
        Then :
            - compute ROI
            - compute obj
            - compute bkg
            - compute boneness

            - use these values to init the GraphCutLinks obj
            - compute the t-links for the source
        Assert
            - the unique t-link values inside roi are 0, 1, 100
                (correspondig to the default lambda)
            - the values outside ROI are empty TODO
            - the values are correctly assigned TODO
        '''
        roi = binary_threshold(image, 200, 50)
        obj = binary_threshold(image, 200, 150)
        bkg = binary_threshold(image, 100, 50)

        bones = Boneness(image, [scale], roi)
        boneness = bones.computeBonenessMeasure()

        link = GraphCutLinks(image, boneness, roi, obj, bkg)
        cost_source = link.tLinkSource()

        obj_arr = itk.GetArrayFromImage(obj)
        roi_arr = itk.GetArrayFromImage(roi)

        assert np.all(cost_source.shape == obj_arr.shape)
        assert np.all(np.unique(cost_source[roi_arr != 0]) == [0, 1, 100])

    @given(random_image_strategy(), st.floats(0.1, 1.5))
    @settings(max_examples=20, deadline=None,
              suppress_health_check=(HC.too_slow, ))
    def test_tLinkSink(self, image, scale) :
        '''
        Given:
            - image
            - scale (single is enough)
        Then :
            - compute ROI
            - compute obj
            - compute bkg
            - compute boneness

            - use these values to init the GraphCutLinks obj
            - compute the t-links for the source
        Assert
            - the unique t-link values inside roi are 0, 1, 100
                (correspondig to the default lambda)
            - the values outside ROI are empty TODO
            - the values are correctly assigned TODO
        '''
        roi = binary_threshold(image, 200, 50)
        obj = binary_threshold(image, 200, 150)
        bkg = binary_threshold(image, 100, 50)

        bones = Boneness(image, [scale], roi)
        boneness = bones.computeBonenessMeasure()

        link = GraphCutLinks(image, boneness, roi, obj, bkg)
        cost_sink = link.tLinkSink()

        obj_arr = itk.GetArrayFromImage(obj)
        roi_arr = itk.GetArrayFromImage(roi)
        bkg_arr = itk.GetArrayFromImage(bkg)

        assert np.all(cost_sink.shape == obj_arr.shape)
        assert np.all(np.unique(cost_sink[roi_arr != 0]) == [0, 1, 100])
        assert np.all(np.unique(cost_sink[obj_arr != 0]) == 0)
        assert np.all(np.unique(cost_sink[bkg_arr != 0]) == 100)


# TODO add test to nLink method
# TODO add test to getLinks method
