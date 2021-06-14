#!/usr/bin/env python
# -*- coding: utf-8 -*-


import itk
import numpy as np

__author__ = ['Biondi Riccardo']
__email__ = ['riccardo.biondi7@unibo.it']

# TODO add healt check and error/exception handling


def cast_image(image, new_pixel_type) :
    '''
    cast the image pixel type to new_pixel_type

    Parameters
    ----------

    image : itk.Image
        image to cast
    new_pixel_type : itk pixel type
        new image pixel type

    Return
    ------

    castedImage : itk image obj
        image with pixel fo new_pixel_type
    '''
    oldPixelType, dimension = itk.template(image)[1]
    newImageType = itk.Image[new_pixel_type, dimension]
    oldImageType = itk.Image[oldPixelType, dimension]
    castImageFilter = itk.CastImageFilter[oldImageType, newImageType].New()
    castImageFilter.SetInput(image)
    castImageFilter.Update()

    return castImageFilter.GetOutput()





def image2array(image) :
    '''
    Return the image array and a dictionary containing the spatial information
    of an itk image obj

    Parameters
    ----------
    image : itk image obj
        image from which get the array
    Return
    ------
    array : np.ndarray
        image array
    info : dic
        dictionary with all the spatial informations
    '''
    # TODO test
    info = get_image_spatial_info(image)
    array = itk.GetArrayFromImage(image)

    return array, info



def array2image(image_array, spatial_info = None) :
    '''
    Convert an image array into an itk image obj. If provided, set also the
    spatial information

    Parameters
    ----------
    image_array : np.ndarray
        image array to convert
    spatial_info : dic
        python dictionary which contains all the image spatial information

    Return
    ------
    image : itk image obj
    '''
    # TODO test
    image = itk.GetImageFromArray(image_array)
    if spatial_info is not None :
        _ = set_image_spatial_info(image, spatial_info)

    return image



def get_image_spatial_info(image) :
    '''
    Return a dict containing the image spatial information
    Parameter
    ---------
    image : itk image obj
    Return
    ------
    info : dict
        dict containing the spatial information
    '''
    # TODO test
    lpr = image.GetLargestPossibleRegion()
    size = lpr.GetSize()
    index = lpr.GetIndex()
    upperIndex = lpr.GetUpperIndex()
    direction = image.GetDirection()
    spacing = image.GetSpacing()
    origin = image.GetOrigin()

    return {'Direction' : direction,
            'Spacing' : spacing,
            'Origin' : origin,
            'Size' : size,
            'Index' : index,
            'Upper Index' : upperIndex}



def set_image_spatial_info(image, info) :
    '''
    Set the image spatial information to info
    Paramter
    --------
    image : itk Image obj
    info : dict
    '''
    # TODO test
    _ = image.SetSpacing(info['Spacing'])
    _ = image.SetOrigin(info['Origin'])
    _ = image.SetDirection(info['Direction'])
    _ = image.GetLargestPossibleRegion().SetSize(info['Size'])
    _ = image.GetLargestPossibleRegion().SetIndex(info['Index'])
    _ = image.GetLargestPossibleRegion().SetUpperIndex(info['Upper Index'])



def get_labeled_leg(leg1, leg2) :
    '''
    The image to work with has only one out of two labeled femur.
    This function get the separed legs qith the corresponding labels and
    return the
    '''
    l1 = itk.GetArrayFromImage(leg1[1])
    l2 = itk.GetArrayFromImage(leg2[1])
    if np.sum(l1) != 0 :

        return leg1

    elif np.sum(l2) != 0 :

        return leg2
    else :
        raise ValueError('No label found')

# TODO apply_mask
# TODO elementwise_multiplication
# TODO image_addition
