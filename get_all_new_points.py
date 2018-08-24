#!/usr/bin/python2

r'''Synopsis:

xxx

xxx

'''

import numpy as np
import numpysane as nps
import scipy.spatial
import argparse
import sys



def parse_args():
    parser = \
        argparse.ArgumentParser(description = __doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--threshold',
                        type=float,
                        default=10,
                        required=False,
                        help='Threshold distance for two points to be considered "close-enough"')

    parser.add_argument('query_points',
                        type=str,
                        help='''A file containing points in the QUERY set. The output will be a subset of
                        this set; those points that are far-enough from points
                        in the MASK set. Points are stored as 'ID x1 y1 x2 y2'
                        ascii values; one point per line. I look ONLY at x1,y1.
                        Lines starting with # are comments. Vnlog input is valid
                        here''')

    parser.add_argument('mask_points',
                        type=str,
                        help='''A file containing points in the MASK set. The output will contain points that
                        are far-enough away from ALL points in this set. Points
                        are stored as 'ID x1 y1 x2 y2' ascii values; one point per
                        line. I look ONLY at x1,y1. Lines starting with # are
                        comments. Vnlog input is valid here''')

    return parser.parse_args()

args = parse_args()

query = np.loadtxt(args.query_points)
mask  = np.loadtxt(args.mask_points)

mask_tree = scipy.spatial.cKDTree(mask[:,1:3])

distances,indices = mask_tree.query(query[:,1:3], distance_upper_bound=args.threshold)

# I report on those points in the query set that have a too-high distance
p = query[np.arange(distances.shape[0])[ np.isinf(distances) ], :]

np.savetxt(sys.stdout, p, fmt='%d %f %f %f %f', header='ID Feat1x Feat1y Feat2x Feat2y')
