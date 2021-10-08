#!/bin/env python

import os
import itk
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob

from FemurSegmentation.IOManager import ImageReader
from FemurSegmentation.IOManager import VolumeWriter
from FemurSegmentation.filters import execute_pipeline
from FemurSegmentation.filters import adjust_physical_space

from FemurSegmentation.metrics import itk_label_overlapping_measures
from FemurSegmentation.metrics import itk_hausdorff_distance
from FemurSegmentation.metrics import itk_distance_map_source_to_target

# %%


def parse_args():

    description = 'Automated CT Femur Segmentation'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--source',
                        dest='source',
                        required=True,
                        type=str,
                        action='store',
                        help='Source Image Filename')
    parser.add_argument('--target',
                         dest='target',
                         required=True,
                         type=str,
                         action='store',
                         help='Target Image Filename')
    parser.add_argument('--output',
                        dest='output',
                        required=True,
                        type=str,
                        action='store',
                        help='output csv in which save the results')
    parser.add_argument('--distance_map',
                        dest='distance_map',
                        required=False,
                        type=str,
                        action='store',
                        help='output filename for the distance map between source and target',
                        default=None)
    args = parser.parse_args()

    return args



def main(source_path, target_path, compute_distance_map=False):

    ImageType = itk.Image[itk.SS, 3]
    reader = ImageReader()

    name = os.path.basename(source_path)

    source = reader(source_path, ImageType)
    target = reader(target_path, ImageType)

    source = adjust_physical_space(source, target, ImageType)

    measures = itk_label_overlapping_measures(source, target)
    _ = measures.Update()

    hd = itk_hausdorff_distance(source, target)
    _ = hd.Update()

    distance_map = None

    if compute_distance_map:
        distance_map = itk_distance_map_source_to_target(source, target)

    dict = {'Patient Name' : [name],
            'Dice Coefficient' : [measures.GetDiceCoefficient()],
            'Jaccard Coefficient' : [measures.GetJaccardCoefficient()],
            'Volume Similarity' : [measures.GetVolumeSimilarity()],
            'Hausdorff Distance' : [hd.GetHausdorffDistance()],
            'Average Hausdorff Distance' : [hd.GetAverageHausdorffDistance()]}

    df = pd.DataFrame.from_dict(dict)

    print('Processed Image: {}'.format(name), flush=True)
    print('Computed Metrics:', flush=True)
    print(df)


    return [df, distance_map]


if __name__ == '__main__':

    args = parse_args()

    print('Source Image: {}'.format(args.source), flush=True)
    print('Target Image: {}'.format(args.target), flush=True)

    compute_distance_map=False

    if args.distance_map is not None:
        compute_distance_map=True

    df, distance_map = main(args.source, args.target, compute_distance_map=compute_distance_map)

    print('Writing the results to {}'.format(args.output), flush=True)

    df.to_csv(args.output, sep=',', index=False)

    if compute_distance_map:
        print('Writing the distance map to {}'.format(args.distance_map))

        writer = VolumeWriter()
        _ = writer(args.distance_map, distance_map)
    print('[DONE]', flush=True)
