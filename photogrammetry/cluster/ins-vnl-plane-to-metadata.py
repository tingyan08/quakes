#!/usr/bin/python3

r'''Convert a 2021-style per-camera INS vnl into raw-convert.py --metadata input

The 2021/06 flights have the plane INS decoded from the .raw headers into
per-camera vnls (gps_data_v2/batch*cam*.vnl), indexed by frame number:

  # frame_index time_sec_since_midnight_GMT latitude longitude altitude_ft heading_deg groundspeed_knots pitch_deg roll_deg interp_lat interp_lon

raw-convert.py --metadata wants the "images-referenced" format that
reference-images-ins writes for the later flights:

  # time serialnumber filename latitude longitude altitude_ft roll_deg pitch_deg heading_deg

Each .raw file in RAW_DIR is matched to its INS row by the frame number at the
start of the filename, like the old raw-convert.py --ins-vnl-plane did. The
output goes to stdout

'''

import sys
import argparse
import os
import glob
import re

parser = argparse.ArgumentParser(description = __doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--first-frame', type=int,
                    help='''If given, ignore frames before this one''')
parser.add_argument('--last-frame', type=int,
                    help='''If given, ignore frames after this one''')
parser.add_argument('ins_vnl',
                    help='''The per-camera INS vnl''')
parser.add_argument('raw_dir',
                    help='''Directory containing the .raw files for this camera''')
args = parser.parse_args()

import vnlog
import numpy as np

x,list_keys,dict_key_index = vnlog.slurp(args.ins_vnl)
x = np.atleast_2d(x)
row_from_frame = { int(f): i for i,f in enumerate(x[:,dict_key_index['frame_index']]) }

print("# time serialnumber filename latitude longitude altitude_ft roll_deg pitch_deg heading_deg")

Nmissing = 0
for f in sorted(glob.glob(f"{args.raw_dir}/*.raw")):
    m = re.match(r'([0-9]+)_[0-9]+_[0-9]+_[0-9]+_SN([0-9]+)\.raw$', os.path.basename(f))
    if m is None:
        continue
    frame = int(m.group(1))
    if args.first_frame is not None and frame < args.first_frame: continue
    if args.last_frame  is not None and frame > args.last_frame:  continue
    if frame not in row_from_frame:
        Nmissing += 1
        continue
    r = x[row_from_frame[frame]]
    print(r[dict_key_index['time_sec_since_midnight_GMT']],
          m.group(2),
          os.path.abspath(f),
          *['%.17g' % r[dict_key_index[k]] for k in ('latitude','longitude','altitude_ft',
                                                  'roll_deg','pitch_deg','heading_deg')])

if Nmissing:
    print(f"## {Nmissing} images in {args.raw_dir} had no INS data, and were skipped",
          file=sys.stderr)
