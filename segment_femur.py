import itk
import argparse
import numpy as np
import matplotlib.pyplot as plt

# third part
from FemurSegmentation.IOManager import ImageReader, VolumeWriter

from FemurSegmentation.utils import image2array, array2image, cast_image
from FemurSegmentation.utils import get_labeled_leg

from FemurSegmentation.filters import binary_threshold
from FemurSegmentation.filters import connected_components
from FemurSegmentation.filters import relabel_components
from FemurSegmentation.filters import execute_pipeline
from FemurSegmentation.filters import iterative_hole_filling
from FemurSegmentation.filters import distance_map
from FemurSegmentation.filters import add
from FemurSegmentation.filters import apply_pipeline_slice_by_slice
from FemurSegmentation.filters import normalize_image_gl
from FemurSegmentation.filters import unsharp_mask
from FemurSegmentation.filters import median_filter
from FemurSegmentation.filters import erode
from FemurSegmentation.filters import invert_binary_image
from FemurSegmentation.filters import opening
from FemurSegmentation.filters import fill_holes_slice_by_slice

from FemurSegmentation.image_splitter import LegImages
from FemurSegmentation.boneness import Boneness
from FemurSegmentation.links import GraphCutLinks

from GraphCutSupport import RunGraphCut


def view(image, idx) :

    arr = itk.GetArrayFromImage(image)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    _ = ax.axis('off')
    _ = ax.imshow(arr[idx], cmap='gray')

    plt.show()


def parse_args() :
    description = 'A GraphCut based framework for the femur segmentation'
    parser = argparse.ArgumentParser(description=description)

    _ = parser.add_argument('--input',
                            dest='input',
                            required=True,
                            type=str,
                            action='store',
                            help='Path to the input image')
    _ = parser.add_argument('--output',
                            dest='output',
                            required=True,
                            type=str,
                            action='store',
                            help='path to the output folder')
    _ = parser.add_argument('--mask_path',
                            dest='mask_path',
                            required=False,
                            type=str,
                            default='')
    args = parser.parse_args()

    return args



def pre_processing(image, scale=[.75], bkg_lower_thr=-100, bkg_upper_thr=300,
                    bkg_bones_low=0.01, bkg_bones_up=0.5, obj_thr=600,
                    obj_bones_thr=0.3):
    '''
    '''

    # find the body region, whic is the one in which the graph will be computed
    # and also the one in which the image will be normalized accordin to mean
    # and standard deviation.
    # Note: voi = volume of interest
    voi = binary_threshold(image=image, upper_thr=3000,
                            lower_thr=-400, out_type=itk.SS)
    eroded = erode(image=voi, radius=4)
    connected = execute_pipeline(connected_components(eroded.GetOutput()))
    relabeled = relabel_components(image=connected)
    voi = binary_threshold(image=relabeled, upper_thr=2,
                            lower_thr=0, out_type=itk.UC)

    # Now normalize the image according to the voxel mean and standard deviation
    # computed in the volume of interest
    #normalized = normalize_image_gl(image=image, roi=voi)
    normalized = image

    # now compute the single scale boneness measure
    boneness = Boneness(image=normalized, scales=scale, roi=voi)
    bones = boneness.computeBonenessMeasure()
    bones, info = image2array(bones)

    # now find the voxels that certainy belong to the tissue, will be used to
    # set the link with the Sink when we will initialize the graph cut
    bkg, info = image2array(normalized)
    cond = (bkg > bkg_lower_thr) & (bkg < bkg_upper_thr) & (bones > bkg_bones_low) & (bones < bkg_bones_up)
    bkg[cond] = 1
    bkg[~cond] = 0
    bkg = array2image(bkg, info)

    # now find the voxels that centraintly belong to the objec(the femur).
    # will be used to set the links to the source when we will initialize the
    # graph cutonda

    obj, info = image2array(normalized)
    cond = (obj > obj_thr) & (bones > obj_bones_thr)
    obj[cond] = 1
    obj[~cond] = 0

    obj = array2image(obj, info)
    obj = cast_image(obj, itk.SS)
    med = median_filter(image=obj, radius=1)
    median = execute_pipeline(med)
    #print("Median Type:{}".format(type(median)), flush=True)

    # now set to zero all the slices corresponding to the hip joint
    #obj = fill_holes_slice_by_slice(image=median, out_type=itk.SS)
    obj = invert_binary_image(image=median)
    CC = itk.ConnectedComponentImageFilter[itk.Image[itk.UC, 2], itk.Image[itk.UC, 2]].New()
    obj = execute_pipeline(apply_pipeline_slice_by_slice(obj, CC))
    obj = binary_threshold(obj, 2, 0, 0, 1)
    # now find slice by sice the connected components
    #obj, info = image2array(obj)
    #obj[4 * (obj.shape[0] // 5) : ] = 0
    #obj = array2image(obj, info)
    obj = cast_image(obj, itk.SS)
    cc = connected_components(obj)
    cc_im = execute_pipeline(cc)
    rel = relabel_components(cc_im)
    obj = binary_threshold(rel, 2, 0, out_type=itk.UC)

    return voi, bkg, obj


def segmentation(image, roi, bkg, obj, scales=[.5, .6],
                sigma=0.25, Lambda=100, bone_ms_thr=0.3, erosion_radius=3):
    '''
    '''
    _, info = image2array(image)
    # first of all compute the multiscale image filter that will be used to
    # compute the per pixel term of the graph cut
    boneness =  Boneness(image=image, scales=scales, roi=roi)
    multiscale_boneness = boneness.computeBonenessMeasure()
    # find the links to initialize the graph cut
    gc_links = GraphCutLinks(image=image,
                            boneness=multiscale_boneness,
                            roi=roi,
                            obj=obj,
                            bkg=bkg,
                            sigma=sigma,
                            Lambda=Lambda,
                            bone_ms_thr=bone_ms_thr)
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
    labeled = array2image(labelIdImage, info) # ??
    #labeled = execute_pipeline(median_filter(labeled, 1))

    # now fill the holes, apply an erosion, find the connected components
    # and get the largest one, which hopefully is the background
    labeled = invert_binary_image(image=labeled)
    CC = itk.ConnectedComponentImageFilter[itk.Image[itk.UC, 2], itk.Image[itk.UC, 2]].New()
    labeled = execute_pipeline(apply_pipeline_slice_by_slice(labeled, CC))
    labeled = binary_threshold(image=labeled, upper_thr=2, lower_thr=0,
                                in_value=0, out_val=1, out_type=itk.SS)
    #labeled = fill_holes_slice_by_slice(image=labeled, out_type=itk.SS)

    erosion = erode(image=labeled, radius=erosion_radius)
    labeled = execute_pipeline(connected_components(erosion.GetOutput(), itk.SS))
    #labeled = execute_pipeline(connected_components(labeled, itk.SS))
    labeled = relabel_components(labeled)
    labeled = binary_threshold(labeled, 2, 0)
    #eroded = execute_pipeline(erode(labeled, 2))



    return labeled



def post_processing(image, obj, roi, bkg, scales=[0.5, 0.6], Lambda=100,
                    sigma=0.25, bone_ms_thr=0.3, opening_radius=3):
    # compute multiscale boneness
    _, info = image2array(image)
    boneness =  Boneness(image=image, scales=scales, roi=roi)
    multiscale_boneness = boneness.computeBonenessMeasure()

    # now initialize the graph cut links
    gc_links = GraphCutLinks(image=image,
                            boneness=multiscale_boneness,
                            roi=roi,
                            obj=obj,
                            bkg=bkg,
                            sigma=sigma,
                            Lambda=Lambda,
                            bone_ms_thr=bone_ms_thr)
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
    labeled = array2image(labelIdImage, info) #

    #labeled = fill_holes_slice_by_slice(labeled, itk.SS)

    # now refine the segmentation by filling the holes and apply an opening
    labeled = invert_binary_image(image=labeled)
    CC = itk.ConnectedComponentImageFilter[itk.Image[itk.UC, 2], itk.Image[itk.UC, 2]].New()
    labeled = execute_pipeline(apply_pipeline_slice_by_slice(labeled, CC))
    labeled = binary_threshold(image=labeled, upper_thr=2, lower_thr=0,
                                in_value=0, out_val=1, out_type=itk.SS)

    open = opening(labeled, radius=opening_radius)
    # now get the largest component
    im = cast_image(open.GetOutput(), itk.SS)
    cc = connected_components(im, itk.SS)
    labeled = execute_pipeline(cc)
    labeled = relabel_components(labeled)
    labeled = binary_threshold(labeled, 2, 0, out_type=itk.UC)

    return labeled

def main(image, sigma=.5, amount=1., threhsold=0.):

    image = execute_pipeline(unsharp_mask(image, sigma=sigma,
                                        amount=amount, threhsold=threhsold))
    roi, bkg, obj = pre_processing(image=image)
    label = segmentation(image=image, roi=roi, bkg=bkg, obj=obj, erosion_radius=2)
    label = post_processing(image=image, obj=label, roi=roi, bkg=bkg)

    return label

# OLD VERSION

#def pre_processing(image, roi_lower_thr=-100,
#                bkg_lower_thr=0.0,
#                bkg_upper_thr=0.5,
#                bkg_bones_low=0.01, # to exclude marrow bones from hard contrains
#                bkg_bones_up=0.5,
#                obj_thr_gl=1.2,
#                obj_thr_bones=0.3,
#                scale=[1.],
#                sigma=1.,
#                amount=1.,
#                thr=0.) :
#    '''
#    Pre process the image and estimate the ROI in which compute the graph,
#    the object and background voxels to set the Source and Sink tLinks
#    '''
#    print("\tI am pre-processing...", flush=True)
#
#    # find the ROI
#    ROI = binary_threshold(image, 3000, roi_lower_thr, out_type=itk.UC)
#
#    # normalize the image inside the ROI
#    normalized = normalize_image_gl(image, ROI)
#    unsharped = execute_pipeline(unsharp_mask(normalized,
#                                              sigma=sigma,
##                                              amount=amount,
#                                              threhsold=thr))
#
# old version
#    bkg = binary_threshold(unsharped,
#                           bkg_upper_thr,
#                           bkg_lower_thr,
#                           out_type=itk.UC)
    # now compute the single scale boneness measure that will be used to
    # determine the object

#    bones = Boneness(unsharped, scale, ROI)
#    boneness = bones.computeBonenessMeasure()
#    boneness, _ = image2array(boneness)

#    obj, info = image2array(unsharped)
#    cond = (obj > obj_thr_gl) & (boneness > obj_thr_bones)
#    obj[cond] = 1
#    obj[~cond] = 0

#    obj = array2image(obj, info)

    # new version
#    bkg, info = image2array(unsharped)
#    cond = (bkg > bkg_lower_thr) & (bkg < bkg_upper_thr) & (boneness > bkg_bones_low) & (boneness < bkg_bones_up)

    # now get the largest connected region, which will be(hopely) the
    # femur region
#    obj = cast_image(obj, itk.SS)
#    med = medain_filter(image=obj, radius=1)
#    cc = connected_components(med.GetOutput())
#    cc_im = execute_pipeline(cc)
#    rel = relabel_components(cc_im)
#    obj = binary_threshold(rel, 2, 0, out_type=itk.UC)

#    boneness = array2image(boneness, info)

#    return unsharped, ROI, bkg, obj


# OLD VERSION

#def segmentation(image, obj, bkg, ROI, scales=[.5, 1.], sigma=.25,
#                Lambda=100, bone_ms_thr=0.2) :

#    _, info = image2array(image)
    # compute multiscale boneness
#    bones = Boneness(image, scales, ROI)
#    boneness = bones.computeBonenessMeasure()
#    # compute links
#    gc_links = GraphCutLinks(image, boneness, ROI, obj, bkg, sigma=sigma,
#                            Lambda=Lambda, bone_ms_thr=bone_ms_thr)
#    cost_sink_flatten, cost_source_flatten, cost_vx, CentersVx, NeighborsVx, _totalNeighbors, costFromCenter, costToCenter = gc_links.getLinks()
    # apply graph cut
#    uint_gcresult = RunGraphCut(gc_links.total_vx,
#                                np.ascontiguousarray(cost_vx, dtype=np.uint32),
#                                np.ascontiguousarray(cost_source_flatten, dtype=np.uint32),
#                                np.ascontiguousarray(cost_sink_flatten, dtype=np.uint32),
#                                _totalNeighbors,
#                                np.ascontiguousarray(CentersVx, dtype=np.uint32),
#                                np.ascontiguousarray(NeighborsVx, dtype=np.uint32),
#                                np.ascontiguousarray(costFromCenter, dtype=np.uint32),
#                                np.ascontiguousarray(costToCenter, dtype=np.uint32))

#    labelIdImage = gc_links.vx_id
#    labelIdImage[gc_links.vx_id != -1] = uint_gcresult
#    labelIdImage[gc_links.vx_id == -1] = 0
#    labelIdImage = np.asarray(labelIdImage, dtype=np.uint8)
#    labeled = array2image(labelIdImage, info)
#    labeled = execute_pipeline(median_filter(labeled, 1))

#    return labeled


#def post_processing(labeled):
#    '''
#    Fill the labeled image. Get only the femur
#    '''

    # get the largest component (femur)
#    filled = cast_image(labeled, itk.US)
#    cc = connected_components(filled, itk.US)
#    cc_im = execute_pipeline(cc)
#    rel = relabel_components(cc_im)
#    filled = binary_threshold(rel, 2, 0, out_type=itk.UC)

#    filler = iterative_hole_filling(filled, max_iter=5, radius=1)
#    pipe = distance_map(filler.GetOutput())
#    dist = execute_pipeline(pipe)
#    dist = binary_threshold(dist, 25, 0, out_type=itk.SS)

#    negative = itk.ConnectedComponentImageFilter[itk.Image[itk.SS, 2],
#                                                itk.Image[itk.SS, 2]].New()

#    negative = apply_pipeline_slice_by_slice(dist, negative)
#    negative = execute_pipeline(negative)
#    negative = binary_threshold(negative, 700, 1)

#    filled = add(filled, negative)

#    filler = iterative_hole_filling(filled, max_iter=10, radius=1)
#    filled = execute_pipeline(filler)

#    filled = cast_image(filled, itk.US)
#    med = median_filter(image=filled, radius=1)
#    cc = connected_components(med.GetOutput(), itk.US)
#    cc_im = execute_pipeline(cc)
#    rel = relabel_components(cc_im)
#    filled = binary_threshold(rel, 2, 0, out_type=itk.UC)

#    return filled


#def main(image) :

#    unsharped, ROI, bkg, obj = pre_processing(image)
#    labeled = segmentation(unsharped, obj, bkg, ROI)
#    labeled = post_processing(labeled)

#    return labeled


if __name__ == '__main__' :

    args = parse_args()

#    path = '/mnt/d/Riccardo/Progetti/Aperti/FemurSegmentation/Data/Formatted/D0130/D0130.nrrd'
    #print('I am reading the image from: {}'.format(args.input), flush=True)

    reader = ImageReader(args.input, itk.Image[itk.F, 3])
    image = reader.read()

    #view(image, 190)

    # this part is because the dataset we are used has only one labeled
    # leg. This allow us to discriminate
    #between the labeled and unlabeled one
    # and process only one part of the image, reducing the computational time.
    if args.mask_path != '' :

        print('Mask Sepcified, I am reading \
              the GT mask from: {}'.format(args.mask_path), flush=True)
        reader = ImageReader(args.mask_path, itk.Image[itk.UC, 3])
        mask = reader.read()

        splitter = LegImages(image, mask)
        leg1, leg2 = splitter.get_legs()
        image, _ = get_labeled_leg(leg1, leg2)

        label = main(image)

        #view(label, 160)

        print('I am writing...', flush=True)

        writer = VolumeWriter(args.output, label)
        _ = writer.write()

        print("Done", flush=True)

    # TODO reconstruct the original image
