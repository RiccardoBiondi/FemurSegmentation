import itk
import numpy as np
import argparse
import os

from FemurSegmentation.utils import image2array, array2image

from FemurSegmentation.automatic_segmentation import cast_img, extract_slice, dummy_info_2d, get_ROI
from FemurSegmentation.automatic_segmentation import erode_2d, dilate_2d, fill_2d, threshold_slice, applyBinaryThresholdFilter
from FemurSegmentation.automatic_segmentation import get_components_2d, filter_components, directionalConnectedComponentFilter
from FemurSegmentation.automatic_segmentation import boneness_thresholds, get_center_knee, get_center_head
from FemurSegmentation.automatic_segmentation import knee_segmentation_pipe, label_head_slice, score_boolean_array
from FemurSegmentation.automatic_segmentation import prepare_exclusion_region, segment, post_processing
from FemurSegmentation.automatic_segmentation import directionalFillHoles
from FemurSegmentation.automatic_segmentation import erode_img, dilate_img

from FemurSegmentation.peak_detection import get_persistence

from FemurSegmentation.IOManager import VolumeWriter, ImageReader

def parse_args():
    parser = argparse.ArgumentParser(description='Automated CT Femur Segmentation', add_help=False)

    parser.add_argument(
        '--input',
        dest='input', required=True, type=str, action='store',
        help='Input directory - All files inside this directory will be processed'
    )
    parser.add_argument(
        '--output',
        dest='output', required=True, type=str, action='store',
        help='Output directory - Outputs will be saved here'
    )
    parser.add_argument(
        '-h', '--help', 
        action='help', default=argparse.SUPPRESS,
        help = """
            Automatic segmentation of femur CT scans (experimental).
            Input deve essere il path a una directory contenente CT scan di femori, già divise a metà nel senso che ogni CT contiene una sola gamba.
            Abbiamo un tool che esegue questa divisione se necessario. Tutti file in quella directory verranno processati.
            Output deve essere il path alla directory in cui per ogni input scan verrà salvata la corrispondente segmentazione 3D.
            Per ogni cosa scrivere a: federico.magnani9@unibo.it
        """
    )
    args = parser.parse_args()
    return args

def slice_identification(data_arr, info_3d):

    # THRESHOLD
    thr = 0.05
    data_thr = np.zeros_like(data_arr)
    for z in range(data_arr.shape[0]):
        data_thr[z,:,:] = threshold_slice(data_arr[z,:,:], thr)
    thr_img = array2image(data_thr, info_3d)
    thr_SS = cast_img(thr_img, itk.F, itk.SS)

    # CONNECTED COMPONENTS
    # Get the connected filled component of each slice
    components_arr = np.zeros([data_thr.shape[0], data_thr.shape[1], data_thr.shape[2], 3])
    for z in range(data_thr.shape[0]):
        filled_eroded = erode_2d(fill_2d(extract_slice(thr_SS, z)), 1)
        planar_components = get_components_2d(filled_eroded)
        component_1 = applyBinaryThresholdFilter(planar_components, 1,1, dim=2)
        component_2 = applyBinaryThresholdFilter(planar_components, 2,2, dim=2)
        component_3 = applyBinaryThresholdFilter(planar_components, 3,3, dim=2)
        components_arr[z,:,:,0] = image2array(component_1)[0]
        components_arr[z,:,:,1] = image2array(component_2)[0]
        components_arr[z,:,:,2] = image2array(component_3)[0]

    # The filter is to keep only the components that are connected also with respect to z
    filtered_components_arr = filter_components(components_arr)
    components_profile = np.sum(np.sum(filtered_components_arr, axis=1), axis=1)

    # SLICE IDENTIFICATION
    # Get the knee slice
    half_slice = len(components_profile)//2
    center_knee = get_center_knee(components_profile)

    # Get the average of the components made with threshold
    averaged = np.zeros_like(filtered_components_arr)
    for z in range(averaged.shape[0])[half_slice:]:
        averaged[z,:,:] = (filtered_components_arr[z-3,:,:]+filtered_components_arr[z-2,:,:]+filtered_components_arr[z-1,:,:]+filtered_components_arr[z,:,:])/4
    average_profile = np.sum(np.sum(averaged, axis=1), axis=1)

    # That's the center slice of femur head
    # If valley is None: only one peak
    # Else, two peaks and we have the end of the first one
    center_head, valley = get_center_head(average_profile)

    if (valley is None):
        valley = center_head + np.argmin(average_profile[center_head:])

    # The top of the femur is set to 20 slices above the identified "center head" position
    #top_head = np.min([average_profile.shape[0]-1, center_head+20])

    obj_half_slice = filtered_components_arr[half_slice,:,:]

    return(half_slice, center_knee, center_head, valley, obj_half_slice)

def create_segmentation(data_arr, high_boneness_arr):

    # Create approximated segmentation
    segmentation = data_arr*high_boneness_arr
    erosion_radius = 1
    thr = 0.1
    for z in range(segmentation.shape[0]):
        segmentation[z,:,:] = np.where(segmentation[z,:,:]>thr*np.max(segmentation[z,:,:]), 1.0, 0.0)
        segmentation_img = array2image(segmentation[z,:,:], dummy_info_2d(segmentation[z,:,:]))
        segmentation_SS = cast_img(segmentation_img, itk.D, itk.SS, dim=2)
        segmentation_reconstructed = erode_2d(dilate_2d(segmentation_SS, 2), 2)
        segmentation_filled = fill_2d(segmentation_reconstructed)
        segmentation_eroded = erode_2d(segmentation_filled, erosion_radius)
        segmentation[z,:,:], _ = image2array(segmentation_eroded)

    # Interpolate
    for z in range(segmentation.shape[0]):
        if (z!=0)&(z+1!=segmentation.shape[0]):
            overlap = segmentation[z-1,:,:]*segmentation[z+1,:,:]
            segmentation[z,:,:] += overlap
            segmentation[z,:,:] = np.where(segmentation[z,:,:]>0, 1.0, 0.0)

    return(segmentation)

def head_identification(center_head, valley, segmentation, half_slice, body_side, obj, bkg):

    # Look for a slice in which it's possible to identify the head with some certainty
    suitable_slice_array = np.array([])

    start_search = center_head-40
    #end_search = np.min([valley+15, segmentation.shape[0]-1])
    end_search = valley
    search_range = end_search-start_search

    dummy_info = dummy_info_2d(segmentation[half_slice,:,:])
    for slice_offset in range(search_range):

        segmentation_img = array2image(segmentation[start_search+slice_offset,:,:], dummy_info)
        segmentation_SS = cast_img(segmentation_img, itk.D, itk.SS, dim=2)
        suitable_slice, obj_head, bkg_head = label_head_slice(segmentation_SS, body_side)

        # Always update, later the wrong ones will be removed
        bkg_head = np.where(bkg_head>0, 1.0, 0.0)
        obj_head = np.where(obj_head>0, 1.0, 0.0)

        obj[start_search+slice_offset,:,:] = obj_head
        bkg[start_search+slice_offset,:,:] = bkg_head

        suitable_slice_array = np.append(suitable_slice_array, suitable_slice)

    score = score_boolean_array(suitable_slice_array)
    if np.sum(score)!=0:
        # Argmax gives the FIRST occurrence of the max, so the smallest z since the offset
        # went from small z to high z
        slice_offset = np.argmax(score)
    else:
        raise ValueError("No slice could be found for segmenting femur head!")
        #with open("/home/PERSONALE/federico.magnani9/femurSegmentation/test_7/log.txt", "a") as myfile:
        #    message = "[Subject "+str(subj)+"] No slice could be found for segmenting femur head: skipping subject."
        #    myfile.write(message)
        #continue

    return(score, slice_offset, start_search, obj, bkg, obj_head)

def get_cut_slice(obj, bkg, current_segmentation_end, segmentation, valley):

    # Get the top-end of femur head from init profile
    init_profile = np.sum(np.sum(obj+bkg, axis=1), axis=1)
    der_head = init_profile[current_segmentation_end+1:] - init_profile[current_segmentation_end:-1]
    len_der_head = len(der_head)
    if len_der_head>2:
        # Smooth the derivative
        for i in range(len_der_head):
            if i==0:
                der_head[i] = (der_head[i]+der_head[i+1]+der_head[i+2])/3
            elif i==(len(der_head)-1):
                der_head[i] = (der_head[i]+der_head[i-1]+der_head[i-2])/3
            else:
                der_head[i] = (der_head[i]+der_head[i-1]+der_head[i+1])/3

        obj_profile = np.sum(np.sum(obj, axis=1), axis=1)
        end_obj = np.argmin(obj_profile[current_segmentation_end:])
        to_minimize = abs(der_head)[:end_obj]
        to_minimize = np.where(to_minimize<5, 0.0, to_minimize)

        min_der = np.argmin(to_minimize)
        cut_slice = current_segmentation_end+min_der
    else:
        cut_slice = segmentation.shape[0]-1

    #
    cut_slice = np.min([valley, cut_slice])

    return(cut_slice)

def firstGC(filename, input_path, output_path):
    print("    First Graph Cut", flush=True)
    print("        Manual initialization...", flush=True)

    # LOAD DATA
    load_path = input_path+'/'+filename+".nrrd"
    reader = ImageReader()
    raw_img = reader(path=load_path, image_type=itk.Image[itk.F, 3])

    raw_arr, info_3d = image2array(raw_img)

    # GET ROI
    ROI_arr, body_side = get_ROI(raw_img)
    data_arr = raw_arr*ROI_arr

    print("        Slice identification...", flush=True)

    half_slice, center_knee, center_head, valley, obj_half_slice = slice_identification(data_arr, info_3d)

    #-----------------------------------------------------------------------------------------------------

    # Head
    high_boneness_img, _ = boneness_thresholds(raw_img, smooth=10, threshold=0)
    high_boneness_arr, _ = image2array(high_boneness_img)

    #-----------------------------------------------------------------------------------------------------

    # SEGMENTATION
    # Knee
    obj_knee, bkg_knee = knee_segmentation_pipe(data_arr, center_knee)

    segmentation = create_segmentation(data_arr, high_boneness_arr)

    #-----------------------------------------------------------------------------------------------------

    print("        Create obj and bkg...", flush=True)

    # MAKE OBJ AND BKG
    obj = np.zeros_like(data_arr)
    bkg = np.zeros_like(data_arr)

    score, slice_offset, start_search, obj, bkg, obj_head = head_identification(center_head, valley, segmentation, half_slice, body_side, obj, bkg)

    obj[start_search+slice_offset+score[slice_offset]:] = 0
    bkg[start_search+slice_offset+score[slice_offset]:] = 0

    start_head_segmentation = start_search+slice_offset
    end_head_segmentation = start_head_segmentation+score[slice_offset]-1

    obj[center_knee,:,:] = obj_knee
    bkg[center_knee,:,:] = bkg_knee

    obj[obj.shape[0]//2,:,:] = obj_half_slice

    obj = np.where(obj>0, 1.0, 0.0)
    bkg = np.where(bkg>0, 1.0, 0.0)

    #-----------------------------------------------------------------------------------------------------

    # EXPAND SEGMENTATION HEAD

    # Phase 1: segment until merging
    bkg_ref = bkg[start_head_segmentation,:,:]
    obj_ref = obj[start_head_segmentation,:,:]
    z = start_head_segmentation
    stop = False
    while (not stop)&(z < obj.shape[0]-1):

        z += 1

        segmentation_img = array2image(segmentation[z,:,:], dummy_info_2d(segmentation[z,:,:]))
        segmentation_SS = cast_img(segmentation_img, itk.D, itk.SS, dim=2)
        planar_components = get_components_2d(segmentation_SS)
        component_1 = applyBinaryThresholdFilter(planar_components, 1,1, dim=2)
        component_2 = applyBinaryThresholdFilter(planar_components, 2,2, dim=2)
        component_3 = applyBinaryThresholdFilter(planar_components, 3,3, dim=2)
        comp_1_arr, _ = image2array(component_1)
        comp_2_arr, _ = image2array(component_2)
        comp_3_arr, _ = image2array(component_3)

        overlap_1_bkg = (np.sum(comp_1_arr*bkg_ref)!=0)
        overlap_2_bkg = (np.sum(comp_2_arr*bkg_ref)!=0)
        overlap_3_bkg = (np.sum(comp_3_arr*bkg_ref)!=0)
        overlap_1_obj = (np.sum(comp_1_arr*obj_ref)!=0)
        overlap_2_obj = (np.sum(comp_2_arr*obj_ref)!=0)
        overlap_3_obj = (np.sum(comp_3_arr*obj_ref)!=0)

        bkg_ref = np.zeros_like(comp_1_arr)
        obj_ref = np.zeros_like(comp_1_arr)

        if (overlap_1_bkg&overlap_1_obj)|(overlap_2_bkg&overlap_2_obj)|(overlap_3_bkg&overlap_3_obj):
            # If there it is a component that belongs both to obj and bkg
            stop = True
        else:
            # Assign label to component
            if overlap_1_bkg:
                bkg[z,:,:] += comp_1_arr
                bkg_ref += comp_1_arr
            if overlap_1_obj:
                obj[z,:,:] += comp_1_arr
                obj_ref += comp_1_arr
            if overlap_2_bkg:
                bkg[z,:,:] += comp_2_arr
                bkg_ref += comp_2_arr
            if overlap_2_obj:
                obj[z,:,:] += comp_2_arr
                obj_ref += comp_2_arr
            if overlap_3_bkg:
                bkg[z,:,:] += comp_3_arr
                bkg_ref += comp_3_arr
            if overlap_3_obj:
                obj[z,:,:] += comp_3_arr
                obj_ref += comp_3_arr

    current_segmentation_end = z

    obj = np.where(obj>0, 1.0, 0.0)
    bkg = np.where(bkg>0, 1.0, 0.0)

    #-----------------------------------------------------------------------------------------------------

    # Phase 2: segment until extinction
    if current_segmentation_end==obj.shape[0]-1:
        # They never merged!
        current_segmentation_end = end_head_segmentation

    z = current_segmentation_end-1

    obj_ref = np.zeros_like(obj_head)
    obj_ref[:,:] = obj[z,:,:]
    ref_area = np.sum(obj_ref)
    min_area = 0.1*ref_area
    while (ref_area > min_area)&(z < obj.shape[0]-1):
        z += 1
        obj_ref *= segmentation[z,:,:]
        obj[z,:,:] = obj_ref
        ref_area = np.sum(obj_ref)

    #-----------------------------------------------------------------------------------------------------

    cut_slice = get_cut_slice(obj, bkg, current_segmentation_end, segmentation, valley)

    #-----------------------------------------------------------------------------------------------------

    # Between cut_slice and cut_slice+10 (if possible) we let graphcut do the job
    # Turn obj into bkg
    roof_slice = np.min([segmentation.shape[0]-1, cut_slice+10])
    bkg[roof_slice:, :,:] += obj[roof_slice:, :,:]
    # Delete object up to the cut
    obj[cut_slice:, :,:] = 0
    # Add roof
    bkg[roof_slice,:,:] = 1

    #-----------------------------------------------------------------------------------------------------

    # EXPAND SEGMENTATION KNEE
    # Until extinciton
    obj_ref = np.zeros_like(obj_knee)
    obj_ref[:,:] = obj_knee[:,:]
    ref_area = np.sum(obj_knee)
    min_area = 0.1*ref_area
    z = center_knee
    while ref_area > min_area:
        z -= 1
        obj_ref *= segmentation[z,:,:]
        obj[z,:,:] = obj_ref
        ref_area = np.sum(obj_ref)

    # MAKE CONDITION
    bkg = np.where(bkg>0, 2.0, 0.0)
    obj = np.where(obj>0, 1.0, 0.0)

    #-----------------------------------------------------------------------------------------------------

    # SAVE INITS

    condition = bkg+obj
    condition_img = array2image(condition, info_3d)

    #writer = VolumeWriter(path=output_path+'/'+filename+".nrrd", image=condition_img)
    #writer.write()

    #-----------------------------------------------------------------------------------------------------

    print("        Running graph-cut...", flush=True)
    # GRAPH-CUT
    ROI, obj_img, bkg_img, info = prepare_exclusion_region(raw_img, array2image(condition, info_3d))
    labeled = segment(image=raw_img, ROI=ROI, obj=obj_img, bkg=bkg_img, info=info)
    labeled_postproc = post_processing(labeled=labeled)

    #-----------------------------------------------------------------------------------------------------

    # Knee postprocessing
    labeled_postproc_2 = directionalFillHoles(
        directionalConnectedComponentFilter(
            labeled_postproc
        ), 2
    )

    # Cut knee
    results_arr, _ = image2array(labeled_postproc_2)
    results_arr = np.where(results_arr!=0, 1.0, 0.0)
    results_profile = np.sum(np.sum(results_arr, axis=1), axis=1)

    peak_knee = np.argmax(results_profile[:half_slice])
    cut_knee = np.argmin(results_profile[:peak_knee])

    results_arr[:cut_knee, :,:] = 0
    results_profile = np.sum(np.sum(results_arr, axis=1), axis=1)

    #-----------------------------------------------------------------------------------------------------

    # Case in which the higher peak is actually below the knee
    idx = len(results_profile)//2
    p = results_profile[:idx]

    persistence = get_persistence(p)
    non_zero_p = np.where(p!=0, 1.0, 0.0)

    peak_1 = np.argsort(non_zero_p*persistence)[::-1][0]
    peak_2 = np.argsort(non_zero_p*persistence)[::-1][1]

    if(p[peak_2] > p[peak_1]/2):
        # Two peaks -> Cut in the valley
        peak_lower = np.min([peak_1, peak_2])
        peak_higher = np.max([peak_1, peak_2])
        cut_knee = peak_lower + np.argmin(results_profile[peak_lower:peak_higher])

    results_arr[:cut_knee, :,:] = 0

    #-----------------------------------------------------------------------------------------------------

    print("        Save results...", flush=True)

    # SAVE RESULTS

    results_img = array2image(results_arr, info_3d)

    writer = VolumeWriter(path=output_path+'/'+filename+"_firstGC_output.nrrd", image=results_img)
    writer.write()

    #-----------------------------------------------------------------------------------------------------

    # UPDATE INITS
    # Since they will be used later by a second run of graphcut

    # Turn obj into bkg
    bottom_slice = np.max([0, cut_knee-10])
    bkg[:bottom_slice, :,:] += results_arr[:bottom_slice:, :,:]
    # Delete object up to the cut
    obj[:cut_knee, :,:] = 0.0
    # Add bottom
    bkg[bottom_slice,:,:] = 2.0

    #-----------------------------------------------------------------------------------------------------

    # SAVE INITS FOR SECOND GRAPHCUTS

    condition = bkg+obj
    condition_img = array2image(bkg+obj, info_3d)

    writer = VolumeWriter(path=output_path+'/'+filename+"_init.nrrd", image=condition_img)
    writer.write()

    print("    First Graph Cut: done\n", flush=True)

def secondGC(filename, input_path, output_path):

    print("    Second Graph Cut", flush=True)

    # Load graphcut output
    reader = ImageReader()
    graphcut_output_img = reader(path=output_path+'/'+filename+"_firstGC_output.nrrd", image_type=itk.Image[itk.SS, 3])

    # Erode graphcut output
    eroded_img = erode_img(graphcut_output_img, 3)
    eroded_arr, _ = image2array(eroded_img)
    #eroded_profile = np.sum(np.sum(eroded_arr, axis=1), axis=1)

    # Find largest connected component
    ImageType = itk.Image[itk.SS, 3]
    connComp_filter = itk.ConnectedComponentImageFilter[ImageType, ImageType].New()
    relabel_filter = itk.RelabelComponentImageFilter[ImageType, ImageType].New()
    relabel_filter.SetSortByObjectSize(True)

    _ = connComp_filter.SetInput(eroded_img)
    _ = connComp_filter.Update()
    _ = relabel_filter.SetInput(connComp_filter.GetOutput())
    _ = relabel_filter.Update()

    threshold_filter = itk.BinaryThresholdImageFilter[ImageType, ImageType].New()
    threshold_filter.SetLowerThreshold(1)
    threshold_filter.SetUpperThreshold(1)
    threshold_filter.SetInput(relabel_filter.GetOutput())
    _ = threshold_filter.Update()

    largest_connComp_img = threshold_filter.GetOutput()
    connComp_arr, _ = image2array(largest_connComp_img)

    #------------------------------------------------------------------------------------

    graphcut_output_arr, _ = image2array(graphcut_output_img)
    graphcut_profile = np.sum(np.sum(graphcut_output_arr, axis=1), axis=1)

    half_idx = graphcut_profile.shape[0]//2
    max_knee_gc = np.argmax(graphcut_profile[:half_idx])
    max_head_gc = half_idx + np.argmax(graphcut_profile[half_idx:])

    #------------------------------------------------------------------------------------

    # Load data
    reader = ImageReader()
    raw_img = reader(path=input_path+'/'+filename+'.nrrd', image_type=itk.Image[itk.F, 3])
    raw_arr, info_3d = image2array(raw_img)

    # Compute mean value inside the segmentation
    segmented_raw_data = raw_arr*eroded_arr
    segmented_raw_profile = np.sum(np.sum(segmented_raw_data, axis=1), axis=1)
    area_roi = np.sum(np.sum(eroded_arr, axis=1), axis=1)

    mean_profile = np.zeros_like(segmented_raw_profile)
    len_mean_profile = len(mean_profile)
    for z in range(len_mean_profile):
        if area_roi[z] != 0:
            mean_profile[z] = segmented_raw_profile[z]/area_roi[z]
        else:
            mean_profile[z] = 0

    #------------------------------------------------------------------------------------

    # Load first graph cut inits
    reader = ImageReader()
    inits_img = reader(path=output_path+'/'+filename+'_init.nrrd', image_type=itk.Image[itk.F, 3])
    inits_arr, _ = image2array(inits_img)
    bkg_init_arr = np.where(inits_arr==2.0, 1.0, 0.0)
    obj_init_arr = np.where(inits_arr==1.0, 1.0, 0.0)
    obj_init_profile = np.sum(np.sum(obj_init_arr, axis=1), axis=1)

    #------------------------------------------------------------------------------------

    idx = len(mean_profile)//2
    p = mean_profile[:idx]

    persistence = get_persistence(p)
    mask_p = np.where(p!=0, 1.0, 0.0)

    second_max_persistence = np.argsort(mask_p*persistence)[::-1][1]
    third_max_persistence = np.argsort(mask_p*persistence)[::-1][2]
    if persistence[third_max_persistence] > 0.75*persistence[second_max_persistence]:
        cut_knee = np.min([second_max_persistence,third_max_persistence])
    else:
        cut_knee = second_max_persistence

    if cut_knee >= max_knee_gc:
        cut_knee = np.nonzero(obj_init_profile)[0][0]

    cut_knee = np.max([0, cut_knee-3])

    #------------------------------------------------------------------------------------

    idx = len(mean_profile)//2
    p = mean_profile[idx:]

    persistence = get_persistence(p)
    mask_p = np.where(p!=0, 1.0, 0.0)

    peak_1 = idx + np.argsort(mask_p*persistence)[::-1][0]
    peak_2 = idx + np.argsort(mask_p*persistence)[::-1][1]

    higher_peak = np.max([peak_1, peak_2])

    # START TEST ALTERNATIVES
    if higher_peak >= max_head_gc:
        cut_head = higher_peak
        alert_on_previous_init = False
    else:
        alert_on_previous_init = True
        start_bone = np.nonzero(graphcut_profile)[0][0]
        middle_bone = np.argmax(mean_profile)
        cut_head = middle_bone + (middle_bone-start_bone)
        cut_head = np.min([cut_head+3, mean_profile.shape[0]-1])
#    peak_3 = idx + np.argsort(mask_p*persistence)[::-1][2]
#    alert_on_previous_init = False
#    if (higher_peak < max_head_gc) | (persistence[peak_3-idx] > persistence[peak_2-idx]/3):
#        alert_on_previous_init = True

#    if not alert_on_previous_init:
#        cut_head = higher_peak
#    else:
#        start_bone = np.nonzero(graphcut_profile)[0][0]
#        middle_bone = np.argmax(mean_profile)
#        cut_head = middle_bone + (middle_bone-start_bone)
#        cut_head = np.min([cut_head+3, mean_profile.shape[0]-1])
    # END TEST ALTERNATIVES

    cut_head = np.min([cut_head+3, mean_profile.shape[0]-1])

    #------------------------------------------------------------------------------------

    # MAKE NEW INITS

    # Cut cut cut
    connComp_arr[:cut_knee, :,:] = 0
    connComp_arr[cut_head:, :,:] = 0

    # Make bkg by dilation difference
    connComp_img = array2image(connComp_arr, info_3d)
    much_dilated_connComp_img = dilate_img(connComp_img, 15)
    less_dilated_connComp_img = dilate_img(connComp_img, 10)

    bkg_init_arr = np.where(inits_arr==2.0, 1.0, 0.0)

    much_dilated_arr, _ = image2array(much_dilated_connComp_img)
    less_dilated_arr, _ = image2array(less_dilated_connComp_img)
    bkg_arr = much_dilated_arr-less_dilated_arr

    # Casting
    bkg_arr = np.where(bkg_arr>0, 1.0, 0.0)
    bkg_init_arr = np.where(bkg_init_arr>0, 1.0, 0.0)
    obj_init_arr = np.where(obj_init_arr>0, 1.0, 0.0)

    # Add inits to bkg
    if not alert_on_previous_init:
        bkg_arr += bkg_init_arr

    # Add new limits to bkg
    upper_limit = np.min([cut_head+1, bkg_arr.shape[0]-1])
    lower_limit = np.max([0, cut_knee-1])
    bkg_arr[upper_limit, :,:] = 1.0
    bkg_arr[lower_limit, :,:] = 1.0

    # MAKE CONDITION
    bkg = np.where(bkg_arr>0, 2.0, 0.0)
    obj = np.where(connComp_arr>0, 1.0, 0.0)
    condition = bkg+obj
    condition_img = array2image(condition, info_3d)

    #------------------------------------------------------------------------------------

    #writer = VolumeWriter(path=output_path+'/'+filename+"_second_init.nrrd", image=condition_img)
    #writer.write()

    #------------------------------------------------------------------------------------

    print("        Running graph-cut...", flush=True)

    # GRAPH-CUT
    ROI, obj, bkg, _ = prepare_exclusion_region(raw_img, condition_img)
    labeled = segment(image=raw_img, ROI=ROI, obj=obj, bkg=bkg, info=info_3d)
    labeled_postproc = post_processing(labeled=labeled)

    labeled_postproc_2 = directionalConnectedComponentFilter(labeled_postproc)
    labeled_postproc_3 = directionalFillHoles(labeled_postproc_2, 2)
    results_arr, _ = image2array(labeled_postproc_3)
    results_arr = np.where(results_arr!=0, 1.0, 0.0)

    #------------------------------------------------------------------------------------

    results_img = array2image(results_arr, info_3d)

    writer = VolumeWriter(path=output_path+'/'+filename+".nrrd", image=results_img)
    writer.write()

    print("Done\n", flush=True)

def cleanup(filename, output_path):
    os.remove(output_path+'/'+filename+"_init.nrrd")
    os.remove(output_path+'/'+filename+"_firstGC_output.nrrd")

def main():
    args = parse_args()

    input_dir_path = args.input
    output_dir_path = args.output

    if not os.path.isdir(input_dir_path):
        raise ValueError("Input path is not a directory")
    if not os.path.isdir(output_dir_path):
        raise Warning("Output path is not a directory")

    input_list = os.listdir(input_dir_path)

    n_inputs = len(input_list)
    estimated_time = 4*n_inputs
    print(str(n_inputs)+" files are going to be processed.\nEstimated time to complete the task: "+str(estimated_time)+" minutes.", flush=True)

    failed_files = []
    for filename_with_extension in input_list:
        filename = filename_with_extension[:-5]

        print("Processing the file: "+filename, flush=True)
        try:
            firstGC(filename, input_dir_path, output_dir_path)
            secondGC(filename, input_dir_path, output_dir_path)
            cleanup(filename, output_dir_path)
        except Exception:
            print("Failure, the processing of file "+filename+" will be delayed", flush=True)
            failed_files.append(filename)
        print("", flush=True)

    n_failed = len(failed_files)
    n_very_failed = 0
    if n_failed>0:
        print(str(n_failed)+" files are going to be processed again.\n", flush=True)
        very_failed_files = []
        for filename in failed_files:
            print("Processing the file: "+filename, flush=True)
            try:
                firstGC(filename, input_dir_path, output_dir_path)
                secondGC(filename, input_dir_path, output_dir_path)
                cleanup(filename, output_dir_path)
            except Exception:
                print("\nFailure, the file "+filename+" cannot be processed. Contact help at federico.magnani9@unibo.it\n", flush=True)
                very_failed_files.append(filename)

        n_very_failed = len(very_failed_files)

    print(str(n_inputs-n_very_failed)+" files have been processed. Results are in directory called: "+output_dir_path, flush=True)
    if n_very_failed>0:
        print(str(n_very_failed)+" files cannot be processed. They are: ", very_failed_files, "\nContact help at federico.magnani9@unibo.it", flush=True)

if __name__=="__main__":
    main()
