import itk
import sys
import numpy as np

from FemurSegmentation.IOManager import ImageReader

from FemurSegmentation.peak_detection import get_persistence

from FemurSegmentation.utils import image2array, array2image, cast_image
from FemurSegmentation.filters import binary_threshold
from FemurSegmentation.filters import execute_pipeline
from FemurSegmentation.filters import erode
from FemurSegmentation.filters import itk_discrete_gaussian
from FemurSegmentation.filters import getSortedPlanaryCC
from FemurSegmentation.filters import applyMultiplicationFilter

from FemurSegmentation.filters import connected_components
from FemurSegmentation.filters import relabel_components
from FemurSegmentation.filters import binary_curvature_flow
from FemurSegmentation.filters import iterative_hole_filling

from FemurSegmentation.boneness import Boneness

from FemurSegmentation.links import GraphCutLinks
path_to_FemurSegmentation_lib = '/home/PERSONALE/federico.magnani9/femurSegmentation/FemurSegmentation/lib/'
sys.path.append(path_to_FemurSegmentation_lib)
from GraphCutSupport import RunGraphCut

#---------------------------------------------------------------------------------------------------------------------

def patient_code(i):
    str_i = str(i)
    if len(str_i)==1:
        str_i = "00"+str_i
    elif len(str_i)==2:
        str_i = "0"+str_i
    return("Pat"+str_i)

def load_img(file_name, path_to_splitted_images):

    load_path = path_to_splitted_images+file_name+'/'+file_name+".nrrd"
    reader = ImageReader()
    img = reader(path=load_path, image_type=itk.Image[itk.F, 3])
    return(img)

def cast_img(img, in_type, out_type, dim=3):

    castImageFilter = itk.CastImageFilter[itk.Image[in_type, dim], itk.Image[out_type, dim]].New()
    castImageFilter.SetInput(img)
    castImageFilter.Update()
    out_img = castImageFilter.GetOutput()

    return(out_img)

def fill_2d(img):

    foreground_value = int(np.max(image2array(img)[0]))

    fillHoles_filter = itk.BinaryFillholeImageFilter[itk.Image[itk.SS, 2]].New()
    _ = fillHoles_filter.SetInput(img)
    _ = fillHoles_filter.SetForegroundValue(foreground_value)
    _ = fillHoles_filter.Update()
    filled_img = fillHoles_filter.GetOutput()

    return(filled_img)

def extract_slice(img, z):
    """
        Util for extracting z-th slice out of a 3D volume (argument 'img'), with respect to the axial direction
    """

    arr_3d, _ = image2array(img)
    arr_2d = arr_3d[z,:,:]
    info_2d = dummy_info_2d(arr_2d)
    slice_img = array2image(arr_2d, info_2d)

    return(slice_img)

def dummy_info_2d(arr_2d):

    info_2d = {
        "Direction": itk.matrix_from_array([[1.0, 0.0], [0.0, 1.0]]),
        "Spacing": itk.Vector[itk.D, 2]([1.0,1.0]),
        "Origin": itk.Point[itk.D, 2]([0.0,0.0]),
        "Size": itk.Size[2]([arr_2d.shape[0],arr_2d.shape[1]]),
        "Index": itk.Index[2]([0,0]),
        "Upper Index": itk.Index[2]([arr_2d.shape[0]-1,arr_2d.shape[1]-1])
    }

    return(info_2d)

def threshold_slice(arr_2d, thr=0.5):
    brightest = arr_2d.max()
    arr_thr = np.where(arr_2d >  thr*brightest, 1, 0)
    return(arr_thr)

def erode_2d(img, radius):

    foreground_value = int(np.max(image2array(img)[0]))

    ImageType = itk.Image[itk.SS, 2]
    StructuringElementType = itk.FlatStructuringElement[2]
    erosionStructuringElement = StructuringElementType.Ball(int(radius))

    ErodeFilterType = itk.BinaryErodeImageFilter[ImageType, ImageType, StructuringElementType]
    erodeFilter = ErodeFilterType.New()
    _ = erodeFilter.SetInput(img)
    _ = erodeFilter.SetKernel(erosionStructuringElement)
    _ = erodeFilter.SetForegroundValue(foreground_value)
    _ = erodeFilter.SetBackgroundValue(0)
    _ = erodeFilter.Update()
    eroded_img = erodeFilter.GetOutput()

    return(eroded_img)

def dilate_2d(img, radius):

    foreground_value = int(np.max(image2array(img)[0]))

    ImageType = itk.Image[itk.SS, 2]
    StructuringElementType = itk.FlatStructuringElement[2]
    dilationStructuringElement = StructuringElementType.Ball(int(radius))

    DilateFilterType = itk.BinaryDilateImageFilter[ImageType, ImageType, StructuringElementType]
    dilateFilter = DilateFilterType.New()
    _ = dilateFilter.SetInput(img)
    _ = dilateFilter.SetKernel(dilationStructuringElement)
    _ = dilateFilter.SetForegroundValue(foreground_value)
    _ = dilateFilter.SetBackgroundValue(0)
    _ = dilateFilter.Update()
    dilated_img = dilateFilter.GetOutput()

    return(dilated_img)

def applyBinaryThresholdFilter(image, upper_thresh, lower_thresh=1, dim=3):
        
    ImageType = itk.Image[itk.SS, dim]
    threshold_filter = itk.BinaryThresholdImageFilter[ImageType, ImageType].New()
    threshold_filter.SetLowerThreshold(lower_thresh)
    
    threshold_filter.SetUpperThreshold(upper_thresh)
    threshold_filter.SetInput(image)
    _ = threshold_filter.Update()
    
    return threshold_filter.GetOutput()

#---------------------------------------------------------------------------------------------------------------------

def get_components_2d(img):

    ImageType = itk.Image[itk.SS, 2]
    connComp_filter = itk.ConnectedComponentImageFilter[ImageType, ImageType].New()
    relabel_filter = itk.RelabelComponentImageFilter[ImageType, ImageType].New()
    relabel_filter.SetSortByObjectSize(True)

    _ = connComp_filter.SetInput(img)
    _ = connComp_filter.Update()
    labeled = connComp_filter.GetOutput()

    _ = relabel_filter.SetInput(labeled)
    _ = relabel_filter.Update()

    return(relabel_filter.GetOutput())

def filter_components(components_arr):

    start_slice = int(components_arr.shape[0]//2)
    femur_z = components_arr[start_slice,:,:,0]
    
    femur_arr = np.zeros_like(components_arr[:,:,:,0])
    femur_arr[start_slice, :,:] = femur_z

    for z in range(start_slice+1,components_arr.shape[0]):

        overlap_comp_1 = np.sum(femur_z*components_arr[z,:,:,0])
        overlap_comp_2 = np.sum(femur_z*components_arr[z,:,:,1])
        overlap_comp_3 = np.sum(femur_z*components_arr[z,:,:,2])

        component_to_keep = np.argmax([overlap_comp_1, overlap_comp_2, overlap_comp_3])

        femur_z = components_arr[z,:,:,component_to_keep]
        femur_arr[z,:,:] = femur_z

    for z_forward in range(start_slice-1):

        z = start_slice-1-z_forward

        overlap_comp_1 = np.sum(femur_z*components_arr[z,:,:,0])
        overlap_comp_2 = np.sum(femur_z*components_arr[z,:,:,1])
        overlap_comp_3 = np.sum(femur_z*components_arr[z,:,:,2])

        component_to_keep = np.argmax([overlap_comp_1, overlap_comp_2, overlap_comp_3])

        femur_z = components_arr[z,:,:,component_to_keep]
        femur_arr[z,:,:] = femur_z

    return(femur_arr)

#---------------------------------------------------------------------------------------------------------------------

def get_center_knee(profile):

    # One peak if the second is this number of times smaller than the first
    threshold_for_peak_definition = 4

    idx = len(profile)//2
    p = profile[:idx]

    persistence = get_persistence(p)

    peak_1 = np.argsort(persistence)[::-1][0]
    peak_2 = np.argsort(persistence)[::-1][1]

    if persistence[peak_2]*threshold_for_peak_definition < persistence[peak_1]:
        # Only one peak -> We take the tallest
        center_knee = peak_1
    else:
        # Two peaks -> We take the one with higher z
        center_knee = np.max([peak_1, peak_2])

    return(center_knee)

def get_center_head(profile):

    # One peak if the second is this number of times smaller than the first
    threshold_for_peak_definition = 3

    idx = len(profile)//2
    p = profile[idx:]

    persistence = get_persistence(p)

    peak_1 = np.argsort(persistence)[::-1][0]
    peak_2 = np.argsort(persistence)[::-1][1]

    if persistence[peak_2]*threshold_for_peak_definition < persistence[peak_1]:
        # Only one peak -> We take the tallest
        center_head = idx + peak_1
        valley = None
    else:
        # Two peaks -> We take the one with smaller z
        first_peak = idx + np.min([peak_1, peak_2])
        second_peak = idx + np.max([peak_1, peak_2])
        center_head = first_peak
        valley = first_peak + np.argmin(profile[first_peak:second_peak])

    return(center_head, valley)

def slice_shape_descriptors(components_img):
    """
        Apply the LabelImageToShapeLabelMapFilter to 'components_image' (2D): it computes many shape descriptors
    """

    ImageType = itk.Image[itk.UC, 2]
    LabelMapType = itk.LabelMap[itk.StatisticsLabelObject[itk.UL,2]]
    shape_filter = itk.LabelImageToShapeLabelMapFilter[ImageType, LabelMapType].New()

    shape_filter.SetInput(components_img)
    _ = shape_filter.Update()
    stats = shape_filter.GetOutput()
    
    return stats

def threshold_array(arr, thr):
    return( np.where(arr>thr*np.max(arr), 1.0, 0.0) )

def knee_score_slice(x_img):
    x_SS = cast_img(x_img, itk.D, itk.SS, dim=2)
    #x_refined = erode_2d(fill_2d(dilate_2d(x_SS, 3)), 3)
    x_refined = fill_2d(x_SS)
    planar_components = get_components_2d(x_refined)
    component_1 = applyBinaryThresholdFilter(planar_components, 1,1, dim=2)
    component_2 = applyBinaryThresholdFilter(planar_components, 2,2, dim=2)

    comp_1_arr, _ = image2array(component_1)
    comp_2_arr, _ = image2array(component_2)

    if np.sum(comp_2_arr!=0):
        position_1 = np.sum(range(comp_1_arr.shape[0])*np.sum(comp_1_arr, axis=1))/np.sum(comp_1_arr)
        position_2 = np.sum(range(comp_2_arr.shape[0])*np.sum(comp_2_arr, axis=1))/np.sum(comp_2_arr)

        if position_1>position_2:

            comp_1_UC = cast_img(component_1, itk.SS, itk.UC, 2)
            comp_2_UC = cast_img(component_2, itk.SS, itk.UC, 2)
            shape_1_descriptors = slice_shape_descriptors(comp_1_UC)
            shape_2_descriptors = slice_shape_descriptors(comp_2_UC)
            
            label_1 = shape_1_descriptors.GetLabels()[0]
            area_1 = shape_1_descriptors.GetLabelObject(label_1).GetPhysicalSize()
            bounding_box_width_1 = shape_1_descriptors.GetLabelObject(label_1).GetBoundingBox().GetSize()[0]
            bounding_box_height_1 = shape_1_descriptors.GetLabelObject(label_1).GetBoundingBox().GetSize()[1]

            label_2 = shape_2_descriptors.GetLabels()[0]
            area_2 = shape_2_descriptors.GetLabelObject(label_1).GetPhysicalSize()
            bounding_box_width_2 = shape_2_descriptors.GetLabelObject(label_2).GetBoundingBox().GetSize()[0]
            bounding_box_height_2 = shape_2_descriptors.GetLabelObject(label_2).GetBoundingBox().GetSize()[1]
            score = area_1*area_2*(area_1/(bounding_box_width_1*bounding_box_height_1))*(area_2/(bounding_box_width_2*bounding_box_height_2))
        else:
            score = 0
    else:
        score = 0

    return(score, comp_1_arr, comp_2_arr)

def knee_segmentation_pipe(data_arr, center_knee):

    avg_size = 20
    comp_1_arr = np.zeros([avg_size, data_arr.shape[1], data_arr.shape[2]])
    comp_2_arr = np.zeros([avg_size, data_arr.shape[1], data_arr.shape[2]])
    score_array = np.array([])
    for offset in range(avg_size):
        z = center_knee-5+offset

        x = threshold_array(data_arr[z,:,:], 0.1)
        x_img = array2image(x, dummy_info_2d(x))

        score, comp_1, comp_2 = knee_score_slice(x_img)

        score_array = np.append(score_array, score)
        comp_1_arr[offset,:,:] = comp_1
        comp_2_arr[offset,:,:] = comp_2

    knee_slice = np.argmax(score_array)
    return(comp_1_arr[knee_slice,:,:], comp_2_arr[knee_slice,:,:])

def smooth_and_threshold(boneness, ROI_img, smooth, threshold):
    """
        ARGS:
            boneness: itk image
            ROI_arr: itk image
            smooth_rad: int
            thr: between 0 and 1
    """
    
    ROI_arr, _ = image2array(ROI_img)

    # Smooth boneness
    boneness_smooth = execute_pipeline(itk_discrete_gaussian(boneness, smooth))
    _, info = image2array(boneness)

    # Threshold boneness (both positive and negative)
    upper_thr_boneness = binary_threshold(array2image(ROI_arr*boneness_smooth, info), upper_thr=1, lower_thr=threshold)
    lower_thr_boneness = binary_threshold(array2image(ROI_arr*boneness_smooth, info), upper_thr=-threshold, lower_thr=-1)

    return(upper_thr_boneness, lower_thr_boneness)

def boneness_thresholds(img, smooth=10, threshold=0.6):

    # Select patient region
    ROI = binary_threshold(img, upper_thr=3000, lower_thr=-400)

    # Compute boneness
    bones = Boneness(img, [0.5,0.75], ROI)
    boneness = bones.computeBonenessMeasure()

    # Erode ROI patient region
    ROI_SS = cast_img(ROI, itk.F, itk.SS)
    ROI_eroded = execute_pipeline(erode(ROI_SS, radius=4))

    # Get extremes values of boneness as masks
    up_boneness_threshold, low_boneness_threshold = smooth_and_threshold(boneness, ROI_eroded, smooth, threshold)

    return(up_boneness_threshold, low_boneness_threshold)

def get_planar_components_2d(img_SS):
    planar_components = get_components_2d(img_SS)
    component_1 = applyBinaryThresholdFilter(planar_components, 1,1, dim=2)
    component_2 = applyBinaryThresholdFilter(planar_components, 2,2, dim=2)
    component_3 = applyBinaryThresholdFilter(planar_components, 3,3, dim=2)
    comp_1_arr, _ = image2array(component_1)
    comp_2_arr, _ = image2array(component_2)
    comp_3_arr, _ = image2array(component_3)
    return(comp_1_arr, comp_2_arr, comp_3_arr)

def order_components_by_j_position(comp_1_arr, comp_2_arr, comp_3_arr):
        
    j_1 = np.sum(range(comp_1_arr.shape[1])*np.sum(comp_1_arr, axis=0))/np.sum(comp_1_arr)
    j_2 = np.sum(range(comp_2_arr.shape[1])*np.sum(comp_2_arr, axis=0))/np.sum(comp_2_arr)
    j_3 = np.sum(range(comp_3_arr.shape[1])*np.sum(comp_3_arr, axis=0))/np.sum(comp_3_arr)

    left_component_idx = np.argmin([j_1,j_2,j_3])
    right_component_idx = np.argmax([j_1,j_2,j_3])
    middle_component_idx = set([0,1,2]).difference(set([left_component_idx,right_component_idx])).pop()
    
    left_component = [comp_1_arr, comp_2_arr, comp_3_arr][left_component_idx]
    right_component = [comp_1_arr, comp_2_arr, comp_3_arr][right_component_idx]
    middle_component = [comp_1_arr, comp_2_arr, comp_3_arr][middle_component_idx]

    return(left_component, middle_component, right_component)

def label_head_slice(segmentation_SS, body_side):

    segm_comp_1_arr, segm_comp_2_arr, segm_comp_3_arr = get_planar_components_2d(segmentation_SS)
    
    area_1 = np.sum(segm_comp_1_arr)
    area_2 = np.sum(segm_comp_2_arr)
    area_3 = np.sum(segm_comp_3_arr)

    if (area_2/area_1 > 0.2)&(area_3/area_1 > 0.02):
    #if (area_2/area_1 > 0.25)&(np.sum(area_3)!=0):

        left_component, middle_component, right_component = order_components_by_j_position(segm_comp_1_arr, segm_comp_2_arr, segm_comp_3_arr)
        area_left = np.sum(left_component)
        area_middle = np.sum(middle_component)
        area_right = np.sum(right_component)

        if body_side=="left":
            if not (area_left<area_middle)&(area_left<area_right):
                obj = left_component
                bkg = right_component+middle_component
                suitable_slice = True
            else:
                obj = np.zeros_like(segm_comp_1_arr)
                bkg = np.zeros_like(segm_comp_1_arr)
                suitable_slice = False
        else:
            if not (area_right<area_middle)&(area_right<area_left):
                obj = right_component
                bkg = left_component+middle_component
                suitable_slice = True
            else:
                obj = np.zeros_like(segm_comp_1_arr)
                bkg = np.zeros_like(segm_comp_1_arr)
                suitable_slice = False
    else:
        obj = np.zeros_like(segm_comp_1_arr)
        bkg = np.zeros_like(segm_comp_1_arr)
        suitable_slice = False

    return(suitable_slice, obj, bkg)

#---------------------------------------------------------------------------------------------------------------------

def prepare_exclusion_region(image, condition):
    '''
    '''
    ROI = binary_threshold(image, upper_thr=3000, lower_thr=-400)

    cond, info = image2array(condition)

    obj = cond.copy()
    obj[cond == 1] = 1
    obj[cond != 1] = 0

    bkg = cond.copy()
    bkg[cond != 2] = 0
    bkg[cond == 2] = 1

    obj = array2image(obj, info)
    bkg = array2image(bkg, info)

    return ROI, obj, bkg, info

def segment(image, ROI, obj, bkg, info):
    '''
    '''
    # Returns an itk image obj
    
    bn = Boneness(image=image, scales=[.5, .75], roi=ROI)
    
    ms_bones = bn.computeBonenessMeasure()
    
    gc_links = GraphCutLinks(image=image,
                            boneness=ms_bones,
                            roi=ROI,
                            obj=obj,
                            bkg=bkg,
                            sigma=.25,
                            Lambda=100,
                            bone_ms_thr=.3)
    cost_sink_flatten, cost_source_flatten, cost_vx, CentersVx, NeighborsVx, _totalNeighbors, costFromCenter, costToCenter = gc_links.getLinks()

    uint_gcresult = RunGraphCut(gc_links.total_vx,
                                np.ascontiguousarray(cost_vx, dtype=np.uint32),
                                np.ascontiguousarray(cost_source_flatten, dtype=np.uint32),
                                np.ascontiguousarray(cost_sink_flatten, dtype=np.uint32),
                                _totalNeighbors,
                                np.ascontiguousarray(CentersVx, dtype=np.uint32),
                                np.ascontiguousarray(NeighborsVx, dtype=np.uint32),
                                np.ascontiguousarray(costFromCenter, dtype=np.uint32),
                                np.ascontiguousarray(costToCenter, dtype=np.uint32))
    labelIdImage = gc_links.vx_id
    labelIdImage[gc_links.vx_id != -1] = uint_gcresult
    labelIdImage[gc_links.vx_id == -1] = 0
    labelIdImage = np.asarray(labelIdImage, dtype=np.uint8)

    labeled = array2image(labelIdImage, info)

    return labeled

def post_processing(labeled):

    # first of all get the largest connected region, to eliminate the spurious
    # points
    labeled = cast_image(labeled, itk.SS)
    cc = execute_pipeline(connected_components(labeled))
    cc = relabel_components(cc)
    lab = binary_threshold(cc, upper_thr=2, lower_thr=0, out_type=itk.F)

    # now apply a surface smoothing
    lab = execute_pipeline(binary_curvature_flow(image=lab))
    lab = binary_threshold(lab, upper_thr=2., lower_thr=.9, out_type=itk.SS)

    # and fill the holes
    lab = iterative_hole_filling(image=lab, max_iter=5, radius=1,
                               majority_threshold=1, bkg_val=0, fgr_val=1)

    return execute_pipeline(lab)

def directionalConnectedComponentFilter(image):
    """
        Perform a post-processing filter on the segmentation based on the fact that:
            On the axial plane cannot be present more than 2 connected components in the same slice
            On the coronal plane cannot be present more than 3 connected components in the same slice
            ON the sagittal plane cannot be present more than 4 connected components in the same slice
        And such components are the largest on their slice.
    """
    
    # Computing directional connected components
    axial_image = getSortedPlanaryCC(image, 2)
    coronal_image = getSortedPlanaryCC(image, 1)
    sagittal_image = getSortedPlanaryCC(image, 0)

    # Pruning extra components
    axial_image = applyBinaryThresholdFilter(axial_image, 2)
    coronal_image = applyBinaryThresholdFilter(coronal_image, 3)
    sagittal_image = applyBinaryThresholdFilter(sagittal_image, 4)

    # Merging directional images
    final_image = applyMultiplicationFilter( applyMultiplicationFilter(axial_image, coronal_image), sagittal_image )
 
    return final_image

#---------------------------------------------------------------------------------------------------------------------

def get_ROI(raw_img):

    # Select patient region
    ROI = binary_threshold(raw_img, upper_thr=3000, lower_thr=-400)
    raw_arr, _ = image2array(raw_img)
    # Erode ROI patient region
    ROI_SS = cast_img(ROI, itk.F, itk.SS)
    ROI_eroded = execute_pipeline(erode(ROI_SS, radius=4))
    # Fill ROI (it has some hole)
    ROI_arr = np.zeros_like(raw_arr)
    for z in range(raw_arr.shape[0]):
        filled = fill_2d(extract_slice(ROI_eroded, z))
        ROI_arr[z,:,:] = image2array(filled)[0]

    ROI_left = ROI_arr[-1,:,:20]
    ROI_right = ROI_arr[-1,:,-20:]
    if np.sum(ROI_left)>np.sum(ROI_right):
        body_side = "right"
    else:
        body_side = "left"

    return(ROI, body_side)

#---------------------------------------------------------------------------------------------------------------------

def directionalFillHoles(image, dimension):
    """
        dimension:
            0: sagittal
            1: coronal
            2: axial

        Fill holes in 2D, slice by slice, along the given direction
    """
    
    ImageType = itk.Image[itk.SS, 2]
    fillHoles_filter = itk.BinaryFillholeImageFilter[ImageType].New()
    
    filter_ = itk.SliceBySliceImageFilter[itk.Image[itk.SS, 3], itk.Image[itk.SS, 3]].New()
    _ = filter_.SetInput(image)
    _ = filter_.SetFilter(fillHoles_filter)
    _ = filter_.SetDimension(dimension)
    _ = filter_.Update()
    image = filter_.GetOutput()
    
    return image

def erode_img(img, radius, dim=3):

    foreground_value = int(np.max(image2array(img)[0]))

    ImageType = itk.Image[itk.SS, dim]
    StructuringElementType = itk.FlatStructuringElement[dim]
    erosionStructuringElement = StructuringElementType.Ball(int(radius))

    ErodeFilterType = itk.BinaryErodeImageFilter[ImageType, ImageType, StructuringElementType]
    erodeFilter = ErodeFilterType.New()
    _ = erodeFilter.SetInput(img)
    _ = erodeFilter.SetKernel(erosionStructuringElement)
    _ = erodeFilter.SetForegroundValue(foreground_value)
    _ = erodeFilter.SetBackgroundValue(0)
    _ = erodeFilter.Update()
    eroded_img = erodeFilter.GetOutput()

    return(eroded_img)

def dilate_img(img, radius, dim=3):

    foreground_value = int(np.max(image2array(img)[0]))

    ImageType = itk.Image[itk.SS, dim]
    StructuringElementType = itk.FlatStructuringElement[dim]
    dilationStructuringElement = StructuringElementType.Ball(int(radius))

    DilateFilterType = itk.BinaryDilateImageFilter[ImageType, ImageType, StructuringElementType]
    dilateFilter = DilateFilterType.New()
    _ = dilateFilter.SetInput(img)
    _ = dilateFilter.SetKernel(dilationStructuringElement)
    _ = dilateFilter.SetForegroundValue(foreground_value)
    _ = dilateFilter.SetBackgroundValue(0)
    _ = dilateFilter.Update()
    dilated_img = dilateFilter.GetOutput()

    return(dilated_img)

def get_largest_conn_comp(img):

    ImageType = itk.Image[itk.SS, 3]
    connComp_filter = itk.ConnectedComponentImageFilter[ImageType, ImageType].New()
    relabel_filter = itk.RelabelComponentImageFilter[ImageType, ImageType].New()
    relabel_filter.SetSortByObjectSize(True)
    threshold_filter = itk.BinaryThresholdImageFilter[ImageType, ImageType].New()
    threshold_filter.SetLowerThreshold(1)
    threshold_filter.SetUpperThreshold(1)

    connComp_filter.SetInput(img)
    _ = connComp_filter.Update()
    relabel_filter.SetInput(connComp_filter.GetOutput())
    _ = relabel_filter.Update()
    threshold_filter.SetInput(relabel_filter.GetOutput())
    _ = threshold_filter.Update()
    conncomp = threshold_filter.GetOutput()

    return conncomp

#---------------------------------------------------------------------------------------------------------------------

def score_boolean_array(boolean_arr):
    accumulator = 0
    there_is_accumulator = False
    len_array = len(boolean_arr)
    
    score = np.array([0]*len(boolean_arr))
    for i in range(len_array):

        if boolean_arr[i]:
            if not there_is_accumulator:
                accumulator = i
                there_is_accumulator = True
                score[accumulator] += 1

            else:
                score[accumulator] += 1

        else:
            if there_is_accumulator:
                there_is_accumulator = False

    return(score)

#---------------------------------------------------------------------------------------------------------------------

