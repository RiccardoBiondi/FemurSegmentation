import os
import sys

from glob import glob

configfile : './config.yml'

base_dir = config['base_dir']
dcm_tag = config['DICOM_TAG']


subject_id = os.listdir(base_dir)


rule all:
    input:
        left = expand(os.path.join(base_dir, "{name}", "{name}_seg_L.nrrd"), name=subject_id),
        right = expand(os.path.join(base_dir, "{name}", "{name}_seg_R.nrrd"), name=subject_id)


rule split_image:
    input:
        in_ = os.path.join(base_dir, "{name}", "_".join(["{name}", dcm_tag]))
    output:
        left = os.path.join(base_dir, "{name}", "{name}_L.nrrd"),
        right = os.path.join(base_dir, "{name}", "{name}_R.nrrd")
    params:
        out = os.path.join(base_dir, "{name}", "{name}")

    shell:
        "python split_image.py --input='{input.in_}' --output='{params.out}'"


rule segment_femur:
    input:
        left = os.path.join(base_dir, "{name}", "{name}_L.nrrd"),
        rigth = os.path.join(base_dir, "{name}", "{name}_R.nrrd")
    output:
        left_out = os.path.join(base_dir, "{name}", "{name}_seg_L.nrrd"),
        rigth_out = os.path.join(base_dir, "{name}", "{name}_seg_R.nrrd")

    params:
        out = os.path.join(base_dir, "{name}", "{name}")
    shell:
        "python run_test_segmentation.py --left='{input.left}' --right='{input.rigth}' --output='{params.out}'"
