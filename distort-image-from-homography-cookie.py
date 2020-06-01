#!/usr/bin/python2

r'''Applies a homography cookie to visualize how correct it is

Usually I do this:

function imbase() { printf 'G%07d' $1; }
i=20056; ~/src_boats/uavsar/distort-image-from-homography-cookie.py `imbase $i`.{JPG,homographycookie}; geeqie `imbase $i`.JPG `imbase $i`_homographymapped.JPG `imbase $(($i+1))`.JPG

'''

import numpy as np
import numpysane as nps
import cv2
import sys
import re

usage = "{} imagefilename_from homographycookie\n"

if len(sys.argv) != 3:
    sys.stderr.write(usage.format(sys.argv[0]))
    sys.exit(1)

m = re.match("(.*)\.([a-zA-Z]{3})$", sys.argv[1])
if m is None:
    sys.stderr.write("Image filename assumed to have a 3-letter extension. Instead got '{}'\n".format(sys.argv[1]))
    sys.exit(1)

filename_base      = m.group(1)
filename_extension = m.group(2)

im = cv2.imread(sys.argv[1],0)

cookie = np.loadtxt(sys.argv[2])
if cookie.ravel().shape[0] != 11:
    sys.stderr.write("Unknown cookie format. Need 11 numbers; 3x3 homography, followed by W H\n")
    sys.exit(1)

homography = cookie.ravel()[:9].reshape(3,3)

H,W = im.shape

# regular grid on the destination image. Shape (H,W,2)
m = nps.mv(nps.cat(*np.meshgrid(np.arange(W), np.arange(H))), 0, -1)

# my homography maps source to destination. I need to invert it for my warp to
# work
homography_inv = np.linalg.inv(homography)

# m is a bunch of column vectors. Shape: (H,W,2,1)
m = nps.dummy(m, -1)

# shape: (H,W,3)
m_homogeneous = (nps.matmult(homography_inv[:,:2], m) + homography_inv[:,(2,)])[..., 0]

# shape: (H,W,2)
m = (m_homogeneous[..., :2] / m_homogeneous[..., (2,)]).astype(np.float32)

# apply the map
im_remapped = cv2.remap(im, m[...,0], m[...,1], cv2.INTER_LINEAR)

filename_out = filename_base + '_homographymapped.' + filename_extension
cv2.imwrite(filename_out, im_remapped)

print "Done! Wrote '{}'".format(filename_out)
