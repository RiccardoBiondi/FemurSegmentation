#!/bin/env python

# base packages
import itk
import numpy as np
import matplotlib.pyplot as plt

# optimization packages
from copy import deepcopy
from skopt import dump as skdump
from skopt import load as skload
from skopt.learning import GaussianProcessRegressor
from skopt.optimizer import base_minimize
from skopt.utils import check_random_state
from skopt.utils import use_named_args
from skopt.space import Real, Integer, Categorical
from skopt import gp_minimize
from skopt import callbacks

# third part librariess
from FemurSegmentation.IOManager import ImageReader
from FemurSegmentation.utils import cast_image, image2array
from FemurSegmentation.metrics import dice_score
from FemurSegmentation.filters import unsharp_mask, execute_pipeline
from segment_femur import pre_processing, segmentation, post_processing

# ███████ ██    ██ ███    ██  ██████ ████████ ██  ██████  ███    ██     ██████  ███████ ███████
# ██      ██    ██ ████   ██ ██         ██    ██ ██    ██ ████   ██     ██   ██ ██      ██
# █████   ██    ██ ██ ██  ██ ██         ██    ██ ██    ██ ██ ██  ██     ██   ██ █████   █████
# ██      ██    ██ ██  ██ ██ ██         ██    ██ ██    ██ ██  ██ ██     ██   ██ ██      ██
# ██       ██████  ██   ████  ██████    ██    ██  ██████  ██   ████     ██████  ███████ ██

global images
global masks


def view(image, idx=0):
    #array, _ = image2array(image)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
    _ = ax.axis('off')
    _ = ax.imshow(image[idx], cmap='gray')


# dictionary which define the parameters generation process

spaces = {"gc_segment" : [Integer(-1000, 0, "identity", name = "roi_lower_thr"),
                          Real(0.0, 0.5, "uniform", name = "bkg_lower_thr"),
                          Real(0.51, 1.2, "uniform", name = "bkg_upper_thr"),
                          Real(0.95, 1.5, "uniform", name="obj_gl_thr"),
                          Integer(25, 1000, "identity", name="Lambda"),
                          Real(0.1, 1.5, "uniform", name="single_scale"),
                          Real(0.1, 2.0, "uniform", name="obj_thr_bones"),
                          Real(0.1, 1.25, "uniform", name="sigma"),
                          Real(0.1, 1.5, "uniform", name="multiscale_start"),
                          Real(0.1, 0.7, "uniform", name="multiscale_step"),
                          Integer(0, 7, "uniform", name="number_of_scales"),
                          Real(0.1, 1.0, "uniform", name="bone_ms_thr"),
                          Real(-1.2, 1.2, "uniform", name="unsharp_thr"),
                          Real(0.1, 2.5, "uniform", name="unsharp_amount"),
                          Real(0.1, 1.7, "uniform", name="unsharp_sigma"),
                          Real(0.001, 0.1, "uniorm", name="bkg_bones_low"),
                          Real(0.2, 0.5, "uniform", name="bkg_bones_up")],

          "double_gc" : [#pre_processing params
                        Real(0.25, 1.0, "uniorm", name="single_scale"),
                        Integer(-400, 0, 'identity', name="bkg_lower_thr"),
                        Integer(100, 200, "identity", name="bkg_upper_thr"),
                        Integer(300, 700, "identity", name="obj_thr"),
                        Real(0.1, 0.99, "uniform", name="obj_bones_thr"),
                        # segmentation params
                        Real(0.1, 1., "uniform", name="seg_multiscale_start"),
                        Real(0.1, 0.7, "uniform", name="seg_multiscale_step"),
                        Integer(1, 7, "uniform", name="seg_number_of_scales"),
                        Integer(50, 100, "identity", name="seg_Lambda"),
                        Real(0.1, 0.9, "uniform", name="seg_bones_ms_thr"),
                        Categorical([1, 2, 3, 4, 5], name="erosion_radius"),
                        # post processing
                        Real(0.1, 1., "uniform", name="post_multiscale_start"),
                        Real(0.1, 0.7, "uniform", name="post_multiscale_step"),
                        Integer(1, 7, "uniform", name="post_number_of_scales"),
                        Integer(50, 100, "identity", name="post_Lambda"),
                        Real(0.1, 0.9, "uniform", name="post_bones_ms_thr"),
                        Categorical([1, 2, 3, 4, 5], name="opening_radius")]}


def get_hip_joint_region(y_pred, y_true):
    '''
    Since the segmentation in the Femur body is usually perfect, this will
    improve the scores even if the segmentation in the hip joint region
    (the one in which we are interested in), is not so good.

    This function aims to return only this region, in order to evaluate the
    scores only here.

    Parameters
    ----------
    y_pred: itk.Image
        Segmentation predicted by the model
    y_true: itk.Image
        grount truth segmentation

    Return
    ------
    pred_hip_joint: np.ndarray
        image array containing the upper 1/3 of y_pred
    true_hip_joint: np.ndarray
        image containing the upper 1/3 of y_true
    '''

    p_array, _ = image2array(y_pred)
    t_array, _ = image2array(y_true)

    p_shape = p_array.shape
    t_shape = t_array.shape

    assert p_shape == t_shape

    p_array = p_array[2 * p_shape[0] // 3 : ]
    t_array = t_array[2 * t_shape[0] // 3 : ]

    return p_array, t_array


def evaluate(y_true, y_pred, COST_AMPLIFIER=100):
    '''
    Evaluate the goodness of segmentation against the ground truth.
    The goodness is evaluated using the dice coeffcient.
    Since during the optimization we want to minimize a cost function,
    this function returns as a result: COST_AMPLIFIER * (1 - dice)
    '''

    pred_joint, true_joint = get_hip_joint_region(y_pred, y_true)
    dice = dice_score(pred_joint, true_joint)

    return COST_AMPLIFIER * (1 - dice)


def run_segmentation(
                        image,
                        single_scale,
                        bkg_lower_thr,
                        bkg_upper_thr,
                        obj_thr,
                        obj_bones_thr,
                        seg_multiscale_start,
                        seg_multiscale_step,
                        seg_number_of_scales,
                        seg_Lambda,
                        seg_bones_ms_thr,
                        erosion_radius,
                        post_multiscale_start,
                        post_multiscale_step,
                        post_number_of_scales,
                        post_Lambda,
                        post_bones_ms_thr,
                        opening_radius):
    image = execute_pipeline(unsharp_mask(image))
    roi, bkg, obj = pre_processing(image=image,
                                               scale=[single_scale],
                                               obj_thr=obj_thr,
                                               obj_bones_thr=obj_bones_thr)

    seg_scales_max = seg_multiscale_start + seg_number_of_scales * seg_multiscale_step
    seg_scales = np.arange(seg_multiscale_start, seg_scales_max, seg_multiscale_step)

    label = segmentation(image=image,
                         roi=roi,
                         bkg=bkg,
                         obj=obj,
                         erosion_radius=erosion_radius,
                         Lambda=seg_Lambda,
                         bone_ms_thr=seg_bones_ms_thr,
                         scales=seg_scales)
    post_scales_max = post_multiscale_start + post_number_of_scales * post_multiscale_step
    post_scales = np.arange(post_multiscale_start, post_scales_max, post_multiscale_step)
    label = post_processing(image=image,
                            obj=label,
                            roi=roi,
                            bkg=bkg)

    return label
#def run_segmentation(image, roi_lower_thr, bkg_lower_thr, bkg_upper_thr,
#                    obj_gl_thr, Lambda, single_scale, obj_thr_bones,
#                    sigma, multiscale_start, multiscale_step, number_of_scales,
#                    unsharp_thr, unsharp_amount, unsharp_sigma, bone_ms_thr,
#                    bkg_bones_low, bkg_bones_up):
#    multiscale_max = multiscale_start + number_of_scales * multiscale_step
#    scales = np.arange(multiscale_start, multiscale_max, multiscale_step)

#    ROI, bkg, obj = pre_processing(image,
#                                roi_lower_thr=roi_lower_thr,
#                                bkg_lower_thr=bkg_lower_thr,
#                                bkg_upper_thr=bkg_upper_thr,
#                                bkg_bones_low=bkg_bones_low,
#                                bkg_bones_up=bkg_bones_up,
#                                obj_thr_gl=obj_gl_thr,
#                                obj_thr_bones=obj_thr_bones,
#                                scale=[single_scale],
#                                amount=unsharp_amount,
#                                sigma=unsharp_sigma,
#                                thr=unsharp_thr)

#    label = segmentation(image,
#                        obj,
#                        bkg,
#                        ROI,
#                        scales=scales,
#                        sigma=sigma,
#                        Lambda=Lambda,
#                        bone_ms_thr=bone_ms_thr)

#    label = post_processing(label)

#    return label


def run_optimization(space_key, old_skf, n_calls,
                     n_random_starts, outfile, init_seed) :

    space = spaces[space_key]

    @use_named_args(space)
    def objective(**params) :
        res = []

        for im, msk in zip(images, masks):
            result = run_segmentation(
                                        image=im,
                                        single_scale=params["single_scale"],
                                        bkg_lower_thr=params["bkg_lower_thr"],
                                        bkg_upper_thr=params["bkg_upper_thr"],
                                        obj_thr=params["obj_thr"],
                                        obj_bones_thr=params["obj_bones_thr"],
                                        seg_multiscale_start=params["seg_multiscale_start"],
                                        seg_multiscale_step=params["seg_multiscale_step"],
                                        seg_number_of_scales=params["seg_number_of_scales"],
                                        seg_Lambda=params["seg_Lambda"],
                                        seg_bones_ms_thr=params["seg_bones_ms_thr"],
                                        erosion_radius=params["erosion_radius"],
                                        post_multiscale_start=params["post_multiscale_start"],
                                        post_multiscale_step=params["post_multiscale_step"],
                                        post_number_of_scales=params["post_number_of_scales"],
                                        post_Lambda=params["post_Lambda"],
                                        post_bones_ms_thr=params["post_bones_ms_thr"],
                                        opening_radius=params["opening_radius"]
            )

            res.append(evaluate(result, msk))

        print("{} +/- {}".format(np.mean(res), np.std(res)), flush=True)
        return np.mean(res)

        #hds = []

        # split the image and the mask
        #for im, msk in zip(images, masks) :

        #    result = run_segmentation(image=im,
        #                            roi_lower_thr=params["roi_lower_thr"],
        #                            bkg_lower_thr=params["bkg_lower_thr"],
        #                            bkg_upper_thr=params["bkg_upper_thr"],
        #                            bkg_bones_low=params["bkg_bones_low"],
        #                            bkg_bones_up=params["bkg_bones_up"],
        #                            obj_gl_thr=params["obj_gl_thr"],
        #                            Lambda=params["Lambda"],
        #                            single_scale=params["single_scale"],
        #                            obj_thr_bones=params["obj_thr_bones"],
        #                            sigma=params["sigma"],
        #                            multiscale_start=params["multiscale_start"],
        #                            multiscale_step=params["multiscale_step"],
        #                            number_of_scales=params["number_of_scales"],
        #                            unsharp_thr=params["unsharp_thr"],
        #                            unsharp_amount=params["unsharp_amount"],
        #                            unsharp_sigma=params["unsharp_sigma"],
        #                            bone_ms_thr=params["bone_ms_thr"])
        #    result = cast_image(result, itk.SS)

        #    hd = dice_score(msk, result)
        #    hds.append(hd)

        #print("res : {} +/- {}".format(1 - np.mean(hds), np.std(hds)))

        #return 100 * (1 - np.mean(hds))

    checkpoint_callback = callbacks.CheckpointSaver(outfile,
                                                    store_objective=False)
    # run from scratch
    if old_skf == "" :

        print("run optimization from scratch", flush=True)

        clsf_gp = gp_minimize(objective,
                            space,
                            acq_func="EI",
                            callback=[checkpoint_callback],
                            n_calls=n_calls,
                            n_random_starts=n_random_starts,
                            random_state=init_seed,
                            noise=1e-10)

    else :
        print("Retrieving old result and carry on the optimization from where \
            it was left")

        old_clsf_gp = skload(old_skf)
        args = deepcopy(old_clsf_gp.spect['args'])
        args['n_calls'] += n_calls
        iters = list(old_clsf_gp.x_iters)
        y_iters = list(old_clsf_gp.func_vals)

        if(isinstance(args['random_state'], np.random.RandomState)):
            args['random_state'] = check_random_state(init_seed)
            # gp_minimize related
        if(isinstance(old_clsf_gp.specs['args']['base_estimator'], GaussianProcessRegressor)):
            args['random_state'].randint(0, np.iinfo(np.int32).max)

        def check_or_opt(params):
            if(len(iters) > 0):
                y = y_iters.pop(0)
                if(params != iters.pop(0)):
                    print("Deviated from expected value, re-evaluating")
                else:

                    return y
            return objective(params)
            args['callback'] = [checkpoint_callback]
            args['func'] = check_or_opt
            clsf_gp = base_minimize(**args)
            clsf_gp.specs['args']['func'] = objective

    return clsf_gp

    # ██████  ██    ██ ███    ██      ██████  ██████  ████████ ██ ███    ███ ██ ███████  █████  ████████ ██  ██████  ███    ██
    # ██   ██ ██    ██ ████   ██     ██    ██ ██   ██    ██    ██ ████  ████ ██    ███  ██   ██    ██    ██ ██    ██ ████   ██
    # ██████  ██    ██ ██ ██  ██     ██    ██ ██████     ██    ██ ██ ████ ██ ██   ███   ███████    ██    ██ ██    ██ ██ ██  ██
    # ██   ██ ██    ██ ██  ██ ██     ██    ██ ██         ██    ██ ██  ██  ██ ██  ███    ██   ██    ██    ██ ██    ██ ██  ██ ██
    # ██   ██  ██████  ██   ████      ██████  ██         ██    ██ ██      ██ ██ ███████ ██   ██    ██    ██  ██████  ██   ████


if __name__ == '__main__':

    images = []
    masks = []

    image_path = '/mnt/d/Riccardo/Progetti/Aperti/FemurSegmentation/Data/Formatted/{}/{}.nrrd'
    gt_path = '/mnt/d/Riccardo/Progetti/Aperti/FemurSegmentation/Data/Formatted/{}/{}_gt.nrrd'
    names = ['D0062', 'D076']
    outfile = '/mnt/d/Riccardo/Progetti/Aperti/FemurSegmentation/Data/opt.pkl'

    space_key = "double_gc"
    old_skf = ""
    n_random_starts = 5
    n_calls = 70
    init_seed = 101

    for name in names :
        # read the images
        reader = ImageReader(image_path.format(name, name), itk.Image[itk.F, 3])
        _ = images.append(reader.read())

        reader = ImageReader(gt_path.format(name, name), itk.Image[itk.SS, 3])
        _ = masks.append(reader.read())

    # compute parameters
    result = run_optimization(space_key=space_key,
                            old_skf="",
                            n_calls=n_calls,
                            n_random_starts=n_random_starts,
                            outfile=outfile,
                            init_seed=init_seed)
    # Save final results
    skdump(result, outfile, store_objective=False)
