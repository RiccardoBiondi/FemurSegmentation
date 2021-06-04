import itk
import numpy as np
from FemurSegmentation.utils import image2array, array2image, cast_image

# TODO add healt check and error/exception handling

__author__ = ['Riccardo Biondi']
__email__  = ['riccardo.biondi7@unibo.it']

# TODO allows a customize InputType, not only retrive it from the current image
# TODO allows to not alwaise set the input when you define a filter. This
# will help with the application slice by slice

def binary_threshold(image, upper_thr, lower_thr,
                     in_value = 1, out_val = 0, out_type = None) :
    '''
    Return a binary image whith in_value where input image voxel value is in
    ]lower_thr, upper_thr[, out_value otherwise.

    Parameters
    ----------
    image : itk.Image or numpy array
        input image in which apply thresholding

    upper_thr : int of float
        upper threshold value

    lower_thr : int or float
        lower threhsold value

    in_value : int of float, dafault : 1
        value used to label the voxel inside ]lower_thr, upper_thr[

    out_value : int or float, defaul : 0
        value used to label the voxel outside [lower_thr, upper_thr]

    out_type : itk pixel type. Default None
        if specified cast the output voxel type to out_type

    '''
    if isinstance(image, type(np.array)) :
        array = image.copy()
        info = None
    else :
        array, info = image2array(image)

    cond = (array > lower_thr) & (array < upper_thr)
    array[cond] = in_value
    array[~cond] = out_val

    thr = array2image(array, info)

    if out_type is not None :
        thr = cast_image(thr, out_type)
    return thr



def threshold(image, upper_thr, lower_thr, outside_value = -1500, out_type = None) :
    '''
    Assign to all the voxels outside [lower_thr, upper_thr] the value : outside_value

    Parameters
    ----------
    image : itk.Image
        image to threshold
    upper_thr : int, float
        upper threshold value
    lower_thr : int, float
        lower threshold value
    outside_value : int, float
        value to assign to the voxels outside the inteval [lower_thr, upper_thr]
    out_type : itk pixel type (e.g. itk.F), defaul None
        if provided, cast the output image to out_type

    Return
    ------
    thr : itk.Image
        thresholded image
    '''
    arr, info = image2array(image)
    cond = (arr < lower_thr) & (arr > upper_thr)
    arr[cond] = outside_value
    thr = array2image(arr, info)
    if out_type is not None :
        thr = cast_image(thr, out_type)

    return thr



def median_filter(image, radius = 1) :
    '''
    '''
    PixelType, Dim = itk.template(image)[1]
    ImageType = itk.Image[PixelType, Dim]

    median_filter = itk.MedianImageFilter[ImageType, ImageType].New()
    _ = median_filter.SetInput(image)
    _ = median_filter.SetRadius(int(radius))

    return median_filter



def connected_components(image, voxel_type = itk.SS) :
    '''
    '''

    ImageType = itk.Image[voxel_type, 3]

    cc = itk.ConnectedComponentImageFilter[ImageType, ImageType].New()
    _ = cc.SetInput(image)

    return cc


def relabel_components(image, offset = 1, out_type = None) :

    array, info = image2array(image)
    max_label = int(array.max())

    labels, labels_counts =  np.unique(array,return_counts=True)
    labels = labels[np.argsort(labels_counts)[::-1]]
    labels0 = labels[labels != 0]
    new_max_label = offset - 1 + len(labels0)
    new_labels0 = np.arange(offset, new_max_label + 1)
    required_type = np.min_scalar_type(new_max_label)
    output_type = np.dtype(array.dtype)

    if np.dtype(required_type).itemsize > np.dtype(array.dtype).itemsize :
        output_type = required_type

    forward_map = np.zeros(max_label + 1, dtype = output_type)
    forward_map[labels0] = new_labels0
    inverse_map = np.zeros(new_max_label + 1, dtype = output_type)
    inverse_map[offset : ] = labels0
    relabeled = forward_map[array]

    result = array2image(relabeled, info)
    if out_type is not None :
        result = cast_image(result, out_type)

    return result



def gaussian_smoothing(image, sigma = 1., normalize_across_scale = False) :
    '''
    Computes the smoothing of an image by convolution with the Gaussian kernels

    Parameters
    ----------
    image : itk.Image
        image to smooth
    sigma : float default : 1.
        standard deviation of the gaussian kernel
    normalize_across_scale : bool dafault : False
        specify if normalize the Gaussian over the scale

    Return
    ------
    filter : itk.SmoothingRecursiveGaussianImageFilter
        smoothing filter not updated
    '''

    # Retrive image pixel type and dimension
    PixelType, Dim = itk.template(image)[1]
    ImageType = itk.Image[PixelType, Dim]

    smooth = itk.SmoothingRecursiveGaussianImageFilter[ImageType, ImageType].New()
    _ = smooth.SetInput(image)
    _ = smooth.SetSigma(sigma)
    _ = smooth.SetNormalizeAcrossScale(normalize_across_scale)

    return smooth



def hessian_matrix(image, sigma = 1., normalize_across_scale = False) :
    '''
    Computes the Hessian matrix of an image by convolution with the Second and
    Cross derivatives of a Gaussian.

    Parameters
    ----------
    image : itk.Image
        image to process
    sigma : float Default: 1.
        standard deviation of the gaussian kernel
    normalize_across_scale : Bool Default: False
        specify if normalize the Gaussian over the scale
    Result
    ------
    hessian : itk.HessianRecursiveGaussianImageFilter
        hessian filter not updated
    '''

    PixelType, Dim = itk.template(image)[1]
    ImageType = itk.Image[PixelType, Dim]

    hess = itk.HessianRecursiveGaussianImageFilter[ImageType].New()
    _ = hess.SetInput(image)
    _ = hess.SetSigma(sigma)
    _ = hess.SetNormalizeAcrossScale(normalize_across_scale)
    return hess



def get_eigenvalues_map(hessian, dimensions = 3, order = 2) :
    '''
    omputes the eigen-values of every input symmetric matrix pixel.

    Parameters
    ----------
    hessian :
        hessian matrix from which compute the eigen values

    dimesions: int Default: 3

    order : int Defaut 2
        specify the rule to use for sorting the eigenvalues:
            1 ascending order
            2 magnitude ascending order
            3 no order
    Return
    ------

    eigen_filter : itk.SymmetricEigenAnalysisImageFilter
        not updated
    '''

    ## filter declaration and new obj memory allocation (using New)

    eigen = itk.SymmetricEigenAnalysisImageFilter[type(hessian)].New()
    # seting of the dedidred arguments with the specified ones

    _ = eigen.SetInput(hessian)
    _ = eigen.SetDimension(dimensions)
    _ = eigen.OrderEigenValuesBy(order)

    return eigen



def execute_pipeline(pipeline) :
    '''
    Execute an itk filter pipeline and return its output

    Parameter
    ---------

    pipeline : itk image filter pipeline

        pipeline to execute

    Return
    ------
    Pipeline Output

    Example
    -------

    >>> from FemurSegmentation.IOManager import ImageReader
    >>> from FemurSegmentation.filters import gaussian_smoothing
    >>> from FemurSegmentation.filters import hessian_matrix
    >>> from FemurSegmentation.filters import get_eigenvalues_map
    >>> from FemurSegmentation.filters import execute_pipeline

    >>> # read the image to process
    >>> reader = ImageReader('path/to/image', itk.Image[itk.SS, 3])
    >>> input_image = reader.read()

    >>> # create the image processing pipeline
    >>> smoothed =  gaussian_smoothing(image)
    >>> hessian = hessian_matrix(smoothed.GetOutput())
    >>> final_step =  get_eigenvalues_map(hessian.GetOutput())

    >>> # now apply he pipeline and get the result
    >>> eigen_map = execute_pipeline(final_step)
    '''

    # TODO add some input check
    # NOTE : still to test!!

    _ = pipeline.Update()

    return pipeline.GetOutput()



def opening(image, radius = 1, bkg = 0, frg = 1) :
    '''
    Apply a Morphological opening on the targhet image

    Parameters
    ----------
    image : itk.Image
        target image
    radius : int
        kernel radius
    bkg : pixel Type
        value to be considered as bkg. default 0
    frg : pixel type
        value to be considered as foreground

    Return
    ------
    opened : itk.Image
        opened image
    '''

    # retrive image input type
    PixelType, Dim = itk.template(image)[1]
    ImageType = itk.Image[PixelType, Dim]

    # define the ball structuring element for the opening
    StructuringElementType = itk.FlatStructuringElement[Dim]
    struct_element = StructuringElementType.Ball(radius)

    # define the opening filter
    opening = itk.BinaryMorphologicalOpeningImageFilter[ImageType,
                                                        ImageType,
                                                        StructuringElementType]
    opening = opening.New()
    _ = opening.SetInput(image)
    _ = opening.SetKernel(struct_element)
    _ = opening.SetForegroundValue(frg)
    _ = opening.SetBackgroundValue(bkg)

    return opening



def iterative_hole_filling(image, max_iter = 25, radius = 10,
                           majority_threshold = 1, bkg_val = 0, fgr_val = 1) :
    '''
    '''
    PixelType, Dim = itk.template(image)[1]
    ImageType = itk.Image[PixelType, Dim]

    filter_ = itk.VotingBinaryIterativeHoleFillingImageFilter[ImageType].New()
    _ = filter_.SetInput(image)
    _ = filter_.SetRadius(int(radius))
    _ = filter_.SetMaximumNumberOfIterations(int(max_iter))
    _ = filter_.SetBackgroundValue(bkg_val)
    _ = filter_.SetForegroundValue(fgr_val)
    _ = filter_.SetMajorityThreshold(majority_threshold)

    return filter_



def distance_map(image, image_spacing = True) :
    '''
    '''
    ImageType = itk.Image[itk.UC, 3]

    distance = itk.DanielssonDistanceMapImageFilter[ImageType, ImageType].New()
    _ = distance.SetUseImageSpacing(image_spacing)
    _ = distance.SetInput(image)
    return distance


# TODO add unshapr mask imgea filter



def add(im1, im2) :

    arr1, info = image2array(im1)
    arr2, _ = image2array(im2)

    res = arr1 + arr2
    res = (res > 0).astype(np.uint8)

    return array2image(res, info)



def apply_pipeline_slice_by_slice(image, pipeline, dimension = 2, out_type = None) :
    '''
    '''
    PixelType, Dim = itk.template(image)[1]
    ImageType = itk.Image[PixelType, Dim]


    filter_ = itk.SliceBySliceImageFilter[ImageType, ImageType].New()
    _ = filter_.SetInput(image)
    _ = filter_.SetFilter(pipeline)
    _ = filter_.SetDimension(dimension)

    return filter_



def label_image2shape_label_map(image,
                                bkg = 0,
                                compute_perimeter = False,
                                compute_feret_diameter = False,
                                compute_oriented_bounding_box = False) :

    PixelType, Dim = itk.template(image)[1]
    ImageType = itk.Image[PixelType, Dim]
    LabelType = itk.StatisticsLabelObject[itk.UL, Dim]
    LabelMapType = itk.LabelMap[LabelType]
    shape = itk.LabelImageToShapeLabelMapFilter[itk.Image[itk.UC, 3], LabelMapType].New()
    _ = shape.SetInput(image)
    _ = shape.SetComputePerimeter(compute_perimeter)
    _ = shape.SetComputeFeretDiameter(compute_feret_diameter)
    _ = shape.SetComputeOrientedBoundingBox(compute_oriented_bounding_box)

    return shape



def region_of_interest(image, region) :

    PixelType, Dim = itk.template(image)[1]
    ImageType = itk.Image[PixelType, Dim]

    roi = itk.RegionOfInterestImageFilter[ImageType, ImageType].New()
    _ = roi.SetInput(image)
    _ = roi.SetRegionOfInterest(region)

    return roi
