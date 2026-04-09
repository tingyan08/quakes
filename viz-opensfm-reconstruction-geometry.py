#!/usr/bin/python3

import sys
import numpy as np
import numpysane as nps
import mrcal
import json
import re


reconstruction = sys.argv[1]

with open(reconstruction, "r") as f:
    reconstruction = json.load(f)

shots = reconstruction[0]['shots']

imagenames = np.array(list(shots.keys()))


rt_cam_enu = np.array([ nps.glue( np.array(shot['rotation']),
                                  np.array(shot['translation']),
                                  axis = -1) \
                        for shot in shots.values()])


mrcal.show_geometry(rt_cam_enu,
                    cameranames = imagenames,
                    wait = True)
