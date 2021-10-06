import itk
import numpy as np
from FemurSegmentation.utils import image2array, array2image, cast_image

# TODO add healt check and error/exception handling

__author__ = ['Riccardo Biondi']
__email__ = ['riccardo.biondi7@unibo.it']

# TODO allows a customize InputType, not only retrive it from the current image
# TODO allows to not alwaise set the input when you define a filter. This
# will help with the application slice by slice


def erode(image, radius=1):
    '''
    '''
    ImageType = itk.Image[itk.SS, 3]
    StructuringElementType = itk.FlatStructuringElement[3]
    structuringElement = StructuringElementType.Ball(int(radius))

    ErodeFilterType = itk.BinaryErodeImageFilter[ImageType, ImageType, StructuringElementType]
    erodeFilter = ErodeFilterType.New()
    erodeFilter.SetInput(image)
    erodeFilter.SetKernel(structuringElement)
    erodeFilter.SetErodeValue(1)

    return erodeFilter

def binary_threshold(image, upper_thr, lower_thr,
                    in_value=1, out_val=0, out_type=None) :
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

    Return
    ------

    thr: itk.Image
        binary thresholded image

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


def itk_threshold(image, upper_thr=None, lower_thr=None):
    '''
    '''
    PixelType, Dimension = itk.template(image)[1]
    ImageType = itk.Image[itk.UC, Dimension]


def itk_threshold_below(image, thr, outside_value=-1024):
    '''
    '''
    PixelType, Dimension = itk.template(image)[1]
    ImageType = itk.Image[PixelType, Dimension]

    thr_filter = itk.ThresholdImageFilter[ImageType].New()
    _ = thr_filter.SetInput(image)
    _ = thr_filter.SetOutsideValue(outside_value)
    _ = thr_filter.ThresholdBelow(thr)

    return thr_filter


def threshold(image, upper_thr, lower_thr, outside_value=-1500, out_type=None):
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


def median_filter(image, radius=1) :
    '''
    '''
    PixelType, Dim = itk.template(image)[1]
    ImageType = itk.Image[PixelType, Dim]

    median_filter = itk.MedianImageFilter[ImageType, ImageType].New()
    _ = median_filter.SetInput(image)
    _ = median_filter.SetRadius(int(radius))

    return median_filter


def connected_components(image, voxel_type=itk.SS) :
    '''
    '''
    ImageType = itk.Image[voxel_type, 3]

    cc = itk.ConnectedComponentImageFilter[ImageType, ImageType].New()
    _ = cc.SetInput(image)

    return cc


def relabel_components(image, offset=1, out_type=None) :

    array, info = image2array(image)
    max_label = int(array.max())

    labels, labels_counts = np.unique(array, return_counts=True)
    labels = labels[np.argsort(labels_counts)[::-1]]
    labels0 = labels[labels != 0]
    new_max_label = offset - 1 + len(labels0)
    new_labels0 = np.arange(offset, new_max_label + 1)
    required_type = np.min_scalar_type(new_max_label)
    output_type = np.dtype(array.dtype)

    if np.dtype(required_type).itemsize > np.dtype(array.dtype).itemsize :
        output_type = required_type

    forward_map = np.zeros(max_label + 1, dtype=output_type)
    forward_map[labels0] = new_labels0
    inverse_map = np.zeros(new_max_label + 1, dtype=output_type)
    inverse_map[offset :] = labels0
    relabeled = forward_map[array]

    result = array2image(relabeled, info)
    if out_type is not None :
        result = cast_image(result, out_type)

    return result


def gaussian_smoothing(image, sigma=1., normalize_across_scale=False) :
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


def hessian_matrix(image, sigma=1., normalize_across_scale=False) :
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


def get_eigenvalues_map(hessian, dimensions=3, order=2) :
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

    # filter declaration and new obj memory allocation (using New)

    eigen = itk.SymmetricEigenAnalysisImageFilter[type(hessian)].New()
    # seting of the dedidred arguments with the specified ones

    _ = eigen.SetInput(hessian)
    _ = eigen.SetDimension(dimensions)
    _ = eigen.OrderEigenValuesBy(order)

    return eigen


def execute_pipeline(pipeline):
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


def opening(image, radius=1, bkg=0, frg=1, out_type=None) :
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
    struct_element = StructuringElementType.Ball(int(radius))

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


def iterative_hole_filling(image, max_iter=25, radius=10,
                           majority_threshold=1, bkg_val=0, fgr_val=1) :
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


def distance_map(image, image_spacing=True) :
    '''
    '''
    ImageType = itk.Image[itk.UC, 3]

    distance = itk.DanielssonDistanceMapImageFilter[ImageType, ImageType].New()
    _ = distance.SetUseImageSpacing(image_spacing)
    _ = distance.SetInput(image)
    return distance


def unsharp_mask(image, sigma=.5, amount=1.0, threhsold=0.0):
    '''
    Initilize the Unsharp masking filter
    Parameters
    ----------
    image: itk.Image
        image to unsharp
    sigma: float

    amout: float

    threshold: float

    Return
    ------

    '''
    ImageType = itk.Image[itk.template(image)[1]]
    um = itk.UnsharpMaskImageFilter[ImageType, ImageType].New()
    _ = um.SetInput(image)
    _ = um.SetSigmas(sigma)
    _ = um.SetAmount(amount)
    _ = um.SetThreshold(threhsold)

    return um


def add(im1, im2) :

    arr1, info = image2array(im1)
    arr2, _ = image2array(im2)

    res = arr1 + arr2
    res = (res > 0).astype(np.uint8)

    return array2image(res, info)


def apply_pipeline_slice_by_slice(image, pipeline, dimension=2, out_type=None):
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
                                bkg=0,
                                compute_perimeter=False,
                                compute_feret_diameter=False,
                                compute_oriented_bounding_box=False) :

    PixelType, Dim = itk.template(image)[1]
    ImageType = itk.Image[PixelType, Dim]
    LabelType = itk.StatisticsLabelObject[itk.UL, Dim]
    LabelMapType = itk.LabelMap[LabelType]
    shape = itk.LabelImageToShapeLabelMapFilter[ImageType, LabelMapType].New()
    _ = shape.SetInput(image)
    _ = shape.SetComputePerimeter(compute_perimeter)
    _ = shape.SetComputeFeretDiameter(compute_feret_diameter)
    _ = shape.SetComputeOrientedBoundingBox(compute_oriented_bounding_box)

    return shape


def region_of_interest(image, region):

    PixelType, Dim = itk.template(image)[1]
    ImageType = itk.Image[PixelType, Dim]

    roi = itk.RegionOfInterestImageFilter[ImageType, ImageType].New()
    _ = roi.SetInput(image)
    _ = roi.SetRegionOfInterest(region)

    return roi


def normalize_image_gl(image, roi=None, label=1):
    '''
    Normalize voxel GL according to their mean and standard deviation.
    If mask is provided, the filter will rescale the GL according of the mean
    ans std deviation of the region specifiend by the mask
    Notice that mask must be binary ([0, 1], ImageType = [itk.UC, ImageDim])
    and must have the same shape of the input image

    Parameter
    ---------
    image: itk.Image
        image to normalize
    roi: itk.Image
        binary image. If provided, will normalize the input image according
        of the mean and std deviation of the specified region
    label: int, default 1
        value of the GL corresponding to the region of interest
    Return
    ------
    normalizer: itk.NormalizeImageFilter
        Normalize image filter, initialized but not updated

    .. note: Will raise ZeroDivisionError if the provided image have constant GL
    '''
    # TODO this implementation is not so good, it must be replaced with a
    # better one
    arr, info = image2array(image)

    if roi is not None:
        roi_arr, _ = image2array(roi)
        # check roi shape
        if roi_arr.shape != arr.shape :
            raise ValueError("roi image shape: {} doesn't match the \
                             image one : {}".format(roi_arr.shape, arr.shape))
        mean = np.mean(arr[roi_arr == label])
        std = np.std(arr[roi_arr == label])
    else :
        mean = np.mean(arr)
        std = np.std(arr)

    # now shif the image and rescale according to std
    arr = np.float32(arr - mean) / np.float32(std)

    return array2image(arr, info)


def invert_binary_image(image, out_type=itk.UC):
    array, info = image2array(image)
    array = (array == 0).astype(np.uint8)
    out = array2image(array, info)

    if out_type is not None:
        out = cast_image(out, out_type)

    return out


def fill_holes_slice_by_slice(image, out_type=itk.SS):
    '''
    This filter is used to fill the holes of the image by found the connected
    components of the complementary of each slice and settin the largest(the bkg)
    to zero.
    '''
    PixelType, Dim = itk.template(image)[1]
    #######print(PixelType)

    # image filter type. The dimension is reduced by 1 because it will be
    # applied slice by slice
    ImageType = itk.Image[PixelType, 2]
    OutputType = itk.Image[out_type, 2]
    cc_filter = itk.ConnectedComponentImageFilter[ImageType, ImageType].New()

    # invert the binary image
    inverse = invert_binary_image(image)
    connected_image = execute_pipeline(apply_pipeline_slice_by_slice(image=inverse, pipeline=cc_filter))
    #now ensure that the largest connected component(image background) is
    # labeled with 1 and remove it
    connected_image = relabel_components(connected_image)
    filled = binary_threshold(image=connected_image, upper_thr=2, lower_thr=1,
                                in_value=0, out_val=1, out_type=out_type)
    return filled


def binary_curvature_flow(image, number_of_iterations=1, frg=1):
    '''
    Make an instance of the curvature flow filter for binary images.

    Parameters
    ----------
    image: itk.Image
        binary image to apply the filter
    number_of_iterations: int
        number of time the fiter is iterated
    frg: PixelType
        value of the foreground voxels
    Return
    ------
    instance: itk.BinaryMinMaxCurvatureFlowImageFilter instance
    '''

    PixelType, Dim = itk.template(image)[1]
    ImageType = itk.Image[PixelType, Dim]

    f = itk.BinaryMinMaxCurvatureFlowImageFilter[ImageType, ImageType].New()
    _ = f.SetInput(image)
    _ = f.SetThreshold(frg)
    _ = f.SetNumberOfIterations(number_of_iterations)

    return f


def distance_map(image, use_image_spacing=True):
    '''
    '''
    PixelType, Dim = itk.template(image)[1]
    ImageType = itk.Image[PixelType, Dim]

    distance = itk.DanielssonDistanceMapImageFilter[ImageType, ImageType].New()
    _ = distance.SetInput(image)
    _ = distance.SetUseImageSpacing(use_image_spacing)

    return distance



def adjust_physical_space(in_image, ref_image, ImageType):
    '''
    '''

    NNInterpolatorType = itk.NearestNeighborInterpolateImageFunction[ImageType,
                                                                     itk.D]
    interpolator = NNInterpolatorType.New()

    TransformType = itk.IdentityTransform[itk.D, 3]
    transformer = TransformType.New()
    _ = transformer.SetIdentity()

    resampler = itk.ResampleImageFilter[ImageType, ImageType].New()
    _ = resampler.SetInterpolator(interpolator)
    _ = resampler.SetTransform(transformer)
    _ = resampler.SetUseReferenceImage(True)
    _ = resampler.SetReferenceImage(ref_image)
    _ = resampler.SetInput(in_image)
    _ = resampler.Update()

    return resampler.GetOutput()



def itk_multiple_otsu_threshold(image, number_of_thresholds=3,
                                histogram_bins=128, out_value=0):
    '''
    Instantiate the itk multiple otsu threshold. The filter is not updated

    Parameters
    ----------
    image: itk.Image
        image from which compute the threshold values
    number_of_thresholds: int
        number of threshold to consider
    histogram_bins: int
        number of histogram bins
    Return
    ------
    multi_otsu: itk.OtsuMultipleThresholdsImageFilter
    '''
    # prepare the image types for filter initialization
    PixelType, Dimension = itk.template(image)[1]
    ImageType = itk.Image[PixelType, Dimension]

    # now initialize the filter
    multi_otsu = itk.OtsuMultipleThresholdsImageFilter[ImageType, ImageType].New()
    _ = multi_otsu.SetInput(image)
    _ = multi_otsu.SetNumberOfHistogramBins(histogram_bins)
    _ = multi_otsu.SetNumberOfThresholds(number_of_thresholds)

    return multi_otsu
