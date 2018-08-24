#!/usr/bin/python2

import numpy as np
import numpysane as nps
import gnuplotlib as gp
import glob
import re
import sys
import cPickle

sys.path[:0] = '/home/dima/src_boats/mrcal','/home/dima/src_boats/deltapose-lite',

import mrcal
from mrcal import utils
from mrcal import poseutils
import deltapose_lite

def save():
    cPickle.dump([points3d,extrinsics], open("/tmp/dump.pickle","w"),protocol=2)
def load():
    return cPickle.load(open("/tmp/dump.pickle","r"))



inputs  = sorted(glob.glob("[0-9][0-9][0-9].vnl"))

observations_point          = np.array((), dtype=float)
indices_point_camera_points = np.array((), dtype=np.int32)

ID_to_index = {}

for icam in xrange(len(inputs)):
    f      = inputs[icam]
    camera = icam

    p   = np.loadtxt(f)
    IDs = p[:,0].astype(np.int32)
    p   = p[:,1:]

    for i in xrange(len(IDs)):
        if IDs[i] in ID_to_index:
            # I've already seen this point. Add just the new camera's
            # observation; the old one is already in the dataset
            pointindex = ID_to_index[IDs[i]]

            indices_point_camera_points = \
                nps.glue(indices_point_camera_points,
                         np.array((pointindex, camera+1),
                                  dtype=np.int32),
                         axis=-2)
            observations_point = \
                nps.glue(observations_point,
                         p[i,2:],
                         axis=-2)

        else:
            # Brand new feature being observed. Add both the old and new
            # observations
            pointindex = len(ID_to_index)
            ID_to_index[IDs[i]] = pointindex

            indices_point_camera_points = \
                nps.glue(indices_point_camera_points,
                         np.array(((pointindex, camera),
                                   (pointindex, camera+1)),
                                  dtype=np.int32),
                         axis=-2)
            observations_point = \
                nps.glue(observations_point,
                         p[i,:].reshape(2,2),
                         axis=-2)

Nfeatures = len(ID_to_index)
Nframes   = len(inputs) + 1 # each input is cam-camnext correspondences





def seed(Ncameras, Nfeatures, distortion_model, intrinsics_data_one):

    intrinsics = (distortion_model, intrinsics_data_one)

    range0 = 13000.0

    def estimate_consecutive_extrinsics(camera):

        p = np.loadtxt("{:03d}.vnl".format(camera))
        IDs = p[:,0].astype(np.int32)
        p   = p[:,1:]
        t10    = np.array((-150,0,-50.0))
        r10    = np.random.rand(3) * 1e-3
        e10    = nps.glue(r10, t10, axis=-1)

        points_knownrange0 = nps.glue(p[:,:2], range0*np.ones((len(p),1),dtype=float), axis=-1)
        points_knownrange1 = p[:,2:]

        stats = deltapose_lite.calibrate_extrinsics(e10, intrinsics, intrinsics,
                                                    None, None,
                                                    None, None,
                                                    points_knownrange0,   points_knownrange1,
                                                    VERBOSE                = False,
                                                    skip_outlier_rejection = True,

                                                    # lightly nudge the ranges
                                                    RANGE_COST__PIXELS_PER_RELRATIO                  = 0.01,

                                                    # any rotations will do
                                                    REGULARIZE__L2_CAMERA_ROT_FACTOR__PIXELS_PER_DEG = 0.0,

                                                    # very lightly favor the initial translations
                                                    REGULARIZE__L2_CAMERA_POS_FACTOR__PIXELS_PER_M   = 0.01)

        # I now have a computed pose. I can thus triangulate, and compute the
        # better-estimate point coordinates
        points3d_here = deltapose_lite. \
            compute_3d_intersection_lindstrom(e10, intrinsics, intrinsics,
                                              p[:,:2], p[:,2:])

        return e10,IDs,points3d_here


    extrinsics = None
    points3d   = np.zeros((Nfeatures, 3), dtype=float)


    points3d_at       = []
    points3d_at_local = []


    for icam0 in xrange(0,Ncameras-1):
        # compute the estimated transform from camera icam0 to camera icam0+1 I
        # want the transform FROM the ref coord system (cam0) TO this camera
        rt10,IDs,points3d_here = estimate_consecutive_extrinsics(icam0)

        # anything behind the camera, flip it to the front
        index = points3d_here[:,2] < 0
        points3d_here[index,:] *= -1

        # anything with a crazy-high range, scale it down
        range_here = np.sqrt(nps.inner(points3d_here,points3d_here))
        range_mean = np.mean(range_here)
        index = range_here > 30000
        if np.any(index):
            points3d_here[index,:] *= 30000./nps.transpose(range_here[index])

        range_mean = np.mean(np.sqrt(nps.inner(points3d_here,points3d_here)))
        points3d_here *= range0/range_mean
        points3d_at_local.append(points3d_here)


        if icam0 == 0:
            # cam0 IS the ref coord system. Just store it
            extrinsics = nps.atleast_dims(rt10, -2)

            # I overwrite past point estimates here. Makes coding simpler
            points3d[ [ID_to_index[i] for i in IDs],: ] = points3d_here
        else:
            extrinsics = nps.glue(extrinsics,
                                  mrcal.poseutils.compose_rt(rt10, extrinsics[-1]),
                                  axis=-2)
            # I overwrite past point estimates here. Makes coding simpler
            points3d_here = \
                mrcal.poseutils.transform_point_rt(mrcal.poseutils.invert_rt(extrinsics[-2]),
                                                   points3d_here)
            points3d[ [ID_to_index[i] for i in IDs],: ] = points3d_here

        points3d_at.append(points3d_here)


    # args = [(points3d_at[i][:,0],points3d_at[i][:,1],points3d_at[i][:,2],dict(legend=str(i))) for i in xrange(len(points3d_at))]
    # gp.plot3d(*args, _with='points')

    # import IPython
    # IPython.embed()
    # sys.exit()


    return extrinsics,points3d




# gp.plot(observations_point[...,0], observations_point[...,1])

# Each "frame" is an instant in time. As far as mrcal is concerned, each is a
# different "camera".

imagersize=(5120,3840)
f = 20000
distortion_model = 'DISTORTION_NONE'
intrinsics_data_one = np.array((f,f, imagersize[0]/2, imagersize[1]/2), dtype=float)
intrinsics_data_all = np.array(list(intrinsics_data_one)*Nframes, dtype=float).reshape(Nframes, 4)



# extrinsics = np.random.rand(Nframes-1, 6)* 1e-3
# points3d   = np.random.rand(Nfeatures, 3)* 1.0
# extrinsics[:,3:] = np.array((135.01428289,-150.85947397,-53.01273416))*nps.transpose(np.arange(len(extrinsics)) + 1)
# points3d[:,2]  += 13000.0

# points3d_seed,extrinsics_seed       = load()
# points3d[:len(points3d_seed),:]     = points3d_seed
# extrinsics[:len(extrinsics_seed),:] = extrinsics_seed

extrinsics, points3d = seed(Nframes, Nfeatures, distortion_model, intrinsics_data_one)



# import IPython
# IPython.embed()

# gp.plot3d(*nps.transpose(points3d), _with='points', wait=1)
# sys.exit()


# I need a 3rd column for the range. These are all "unknown", except the first
# that I set to a ballpark "right" value. This is the only thing that gives my
# problem scale
observations_point = nps.glue(observations_point,
                              -np.ones((len(observations_point),1),dtype=np.int32),
                              axis=-1)
observations_point[0,2] = 13000.0 # I have ONE reference range


stats = \
mrcal.optimize(intrinsics_data_all, extrinsics,
               None, points3d,
               None, None,
               observations_point, indices_point_camera_points,
               distortion_model,
               do_optimize_intrinsic_core        = False,
               do_optimize_intrinsic_distortions = False,
               skip_outlier_rejection            = True,
               VERBOSE                           = True)
d = extrinsics[1:,3:]-extrinsics[0:-1,3:]
print "I should get ~ 400 m/frame. Got {}".format(np.sqrt(nps.inner(d,d)))

# plot = utils.visualize_solution(intrinsics_data_all, extrinsics,
#                                 None, points3d,
#                                 None, None,
#                                 observations_point, indices_point_camera_points,
#                                 distortion_model,
#                                 axis_scale=100.0)

p = mrcal.project(points3d, distortion_model, intrinsics_data_one)
gp.plot(p[:,0],p[:,1], points3d[:,2], rgbimage='000.jpg', _with='points pt 7 ps 1 palette',
        square=1, tuplesize=3, set=['xrange [*:*] noextend', 'yrange [*:*] reverse noextend'],
        title='UAVSAR preliminary tracking results. {} tracked features over {} frames'.format(Nfeatures,Nframes),
        cbrange=[10000,20000],
        hardcopy='UAVSAR_preliminary_tracking_results_{}features_{}frames.pdf'.format(Nfeatures,Nframes)
)

# points3d_zeromean = points3d - np.mean(points3d, axis=-2)
# l,v = np.linalg.eig(nps.matmult(nps.transpose(points3d_zeromean),points3d_zeromean))
# gp.plot(*nps.transpose(nps.matmult(points3d_zeromean,v)), _with='points palette pt 7 ps 2', tuplesize=3)

import IPython
IPython.embed()
