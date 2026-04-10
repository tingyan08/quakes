#!/usr/bin/python3

r'''Visualize OpenSfM reconstruction.json geometry'''

import sys
import argparse
import re
import os

def parse_args():

    parser = \
        argparse.ArgumentParser(description = __doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--image-select-regex',
                        help='''If given, only display images matching this
                        regex. If omitted, show all the images''')
    parser.add_argument('reconstruction',
                        help='''The reconstruction.json file''')


    return parser.parse_args()

args  = parse_args()

import numpy as np
import numpysane as nps
import mrcal
import json


with open(args.reconstruction, "r") as f:
    reconstruction = json.load(f)

shots = reconstruction[0]['shots']

imagenames = np.array(list(shots.keys()))


rt_cam_enu = np.array([ nps.glue( np.array(shot['rotation']),
                                  np.array(shot['translation']),
                                  axis = -1) \
                        for shot in shots.values()])


if args.image_select_regex is not None:
    mask = np.array([args.image_select_regex in s for s in imagenames], dtype=bool)
    imagenames = imagenames[mask]
    rt_cam_enu = rt_cam_enu[mask]

mrcal.show_geometry(rt_cam_enu,
                    cameranames = imagenames,
                    wait = True)
