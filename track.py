#!/usr/bin/python2

import numpy as np
import numpysane as nps
import gnuplotlib as gp
import glob
import re
import sys
import cPickle
import subprocess
import re

sys.path[:0] = '/home/dima/src_boats/mrcal','/home/dima/src_boats/deltapose-lite',

import mrcal
from mrcal import utils
from mrcal import poseutils
import deltapose_lite

def save():
    cPickle.dump([points3d,extrinsics], open("/tmp/dump.pickle","w"),protocol=2)
def load():
    return cPickle.load(open("/tmp/dump.pickle","r"))



latlon_from_exif = False
inputs        = sorted(glob.glob("RecordedImage_SP-20000C-USB__[0-9][0-9][0-9].vnl"))
range0        = 13000
range_nominal = 13000
range_max     = 20000
imagersize  = (5120,3840)
focallength = 10000
image_extension = 'tif'

# latlon_from_exif = True
# inputs        = sorted(glob.glob("G[0-9][0-9][0-9][0-9][0-9][0-9][0-9].vnl"))[:9]
# range0        = 20
# range_nominal = 100
# range_max     = 2000
# imagersize  = (4000,3000)
# focallength = 10000
# image_extension = 'JPG'

def track_filename(i):
    return inputs[i]
def image_filename(i):
    return inputs[i][:-3] + image_extension

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




def latlon_from_file(filename):
    r'''Returns a lat,lon numpy array, reading EXIF info from a (presumably) .jpg

    Note that this includes a very crude parser that assumes we're in the NW
    quad-sphere

    '''

    text = subprocess.check_output(('exiftool', '-p', '$gpslatitude $gpslongitude', filename),
                                   shell=False, stderr=subprocess.PIPE)
    m = re.match("([0-9.-]+) deg ([0-9.-]+)' ([0-9.-]+)\" N ([0-9.-]+) deg ([0-9.-]+)' ([0-9.-]+)\" W",
                 text)

    lat =   float(m.group(1))+(float(m.group(2))+float(m.group(3))/60)/60
    lon = -(float(m.group(4))+(float(m.group(5))+float(m.group(6))/60)/60)

    return np.array((lat,lon))

def position_fromlatlon(latlon0, latlon, cos_lat0 = None):
    r'''Returns the position (in m) of two latlons relative to each other

    I assume a locally flat tangent plane and a spherical Earth'''

    # lon=th, lat=ph. Earth coord system: [lon=0, lon=0, north]
    # UP = [cos(th)cos(ph), sin(th)cos(ph), sin(ph)]
    # v = R UP
    #
    # I can get d = v1-v0, I then project to
    # RIGHT   = [-sin(th),        cos(th),        0]
    # FORWARD = [-cos(th)sin(ph),-sin(th)sin(ph), cos(ph)]
    #
    # d/R = [cos(th1)cos(ph1), sin(th1)cos(ph1), sin(ph1)] -
    #       [cos(th0)cos(ph0), sin(th0)cos(ph0), sin(ph0)]
    #
    # proj(d,RIGHT) = R* ( (ct1cp1-ct0cp0) * -st0 +
    #                      (st1cp1-st0cp0) *  ct0 +
    #                      0 )
    #               = R* (cp1* (ct0st1 - st0ct1) + cp0*(st0ct0- ct0st0)) =
    #               = R * cp1*sin(dt)
    #
    # proj(d,FORWARD) = R* ( (ct1cp1-ct0cp0) * -ct0sp0 +
    #                        (st1cp1-st0cp0) * -st0sp0 +
    #                        (sp1   -sp0   ) * cp0
    #                 = R* (ct0ct0cp0sp0 - ct1cp1ct0sp0 + st0st0cp0sp0 - st1cp1st0sp0 + sp1cp0 - sp0cp0)
    #                 = R* (cp0sp0 - ct1cp1ct0sp0 - st1cp1st0sp0 + sp1cp0 - sp0cp0)
    #                 = R* ( - ct1cp1ct0sp0 - st1cp1st0sp0 + sp1cp0)
    #                 = R* ( -cp1sp0 cos(dt) + sp1cp0)
    #                 = R* ( -cp1sp0 ( cos(dt)) + sp1cp0)
    # cos(dt) ~ 1 ->
    # proj(d,FORWARD) ~ R* ( -cp1sp0 + sp1cp0)
    #                 = R* sin(dp)
    #
    # locally, cp1 = cos(lat1) is about constant. I use cp0 instead, looking at
    # the reference
    if cos_lat0 is None:
        cos_lat0 = np.cos(latlon0[0]*np.pi/180.0)
    Rearth = 6371.0e3
    sindp = np.sin((latlon[0]-latlon0[0])*np.pi/180.0)
    sindt = np.sin((latlon[1]-latlon0[1])*np.pi/180.0)

    return Rearth * np.array((cos_lat0*sindt, sindp))

# latlon0 = latlon_from_file('G0020009.JPG')
# cos_lat0 = np.cos(np.pi/180.0 * latlon0[0])
# for i in range(20009,20200):
#     filename = 'G00{}.JPG'.format(i)
#     latlon = latlon_from_file(filename)
#     xy = position_fromlatlon(latlon0, latlon, cos_lat0)
#     print xy[0],xy[1]
# sys.exit()




if latlon_from_exif:
    latlon0  = latlon_from_file(image_filename(0))
    cos_lat0 = np.cos(np.pi/180.0 * latlon0[0])
    xy_prev  = np.zeros((2,),dtype=float)
    t10_prev = None

def seed(Ncameras, Nfeatures, distortion_model, intrinsics_data_one):

    intrinsics = (distortion_model, intrinsics_data_one)

    def estimate_consecutive_extrinsics(camera):
        p = np.loadtxt(track_filename(camera))
        IDs = p[:,0].astype(np.int32)
        p   = p[:,1:]


        # get translation seed from exif
        if latlon_from_exif:
            # t10    = np.array((-150,0,-50.0))

            # Here I use the exif in the images to seed the positions. These come
            # out in a GLOBAL coord system: x = "east", y = "north". But I need
            # these in the camera-0 coord system. I don't have that pose, so I
            # arbitrarily pick a rotation
            th = 330. * np.pi/180.0
            cth = np.cos(th)
            sth = np.sin(th)
            R = np.array(((cth,sth),(-sth,cth)))
            global xy_prev
            global t10_prev
            # this is a hack
            try:

                latlon = latlon_from_file(image_filename(camera+1))
                xy     = position_fromlatlon(latlon0, latlon, cos_lat0)
                dxy    = xy_prev - xy
                t10      = nps.glue(nps.matmult(dxy,R).ravel(), 0, axis=-1) # dz = 0 in the seed.
                t10_prev = t10
                xy_prev  = xy
            except:
                t10      = t10_prev

            r10    = np.random.rand(3) * 1e-3
            e10    = nps.glue(r10, t10, axis=-1)

        else:
            r10    = np.random.rand(3) * 1e-3
            t10    = np.random.rand(3) * 1e0
            t10[0] = -400
            e10    = nps.glue(r10, t10, axis=-1)


        print "running deltapose with {} points".format(p.shape[0])

        # # I set the range of ONE point
        # points_unknownrange0 = p[1:,:2]
        # points_unknownrange1 = p[1:,2:]
        # points_knownrange0 = nps.glue(p[(0,),:2], range0, axis=-1)
        # points_knownrange1 = p[(0,),2:]


        # All ranges are unknown
        points_unknownrange0 = p[:,:2]
        points_unknownrange1 = p[:,2:]

        stats = deltapose_lite.calibrate_extrinsics(e10, intrinsics, intrinsics,
                                                    points_unknownrange0,   points_unknownrange1,
                                                    None, None,
                                                    None, None,
                                                    VERBOSE                = False,
                                                    skip_outlier_rejection = True,

                                                    # lightly nudge the ranges
                                                    RANGE_COST__PIXELS_PER_RELRATIO                  = 0.01,

                                                    # any rotations will do
                                                    REGULARIZE__L2_CAMERA_ROT_FACTOR__PIXELS_PER_DEG = 0.0,

                                                    # any translation will do
                                                    REGULARIZE__L2_CAMERA_POS_FACTOR__PIXELS_PER_M   = 0.01)

        # I now have a computed pose. I can thus triangulate, and compute the
        # better-estimate point coordinates
        points3d_here = deltapose_lite. \
            compute_3d_intersection_lindstrom(e10, intrinsics, intrinsics,
                                              p[:,:2], p[:,2:])

        # import IPython
        # IPython.embed()
        # sys.exit()


        # # reprojection test:
        # mrcal.projections.project(points3d_here,intrinsics) - p[:,:2]
        # mrcal.projections.project(mrcal.poseutils.transform_point_rt(e10,points3d_here),intrinsics) - p[:,2:]

        return e10,IDs,points3d_here


    extrinsics = None

    # all points in the coord system of camera0
    points3d   = np.zeros((Nfeatures, 3), dtype=float)

    # 3d points common to views icam0,icam0+1, in the coord system of icam0
    # (each one in a different coord system) There will be Ncameras-1 of these
    points3d_at_local = []

    # 3d points common to views icam0,icam0+1, in the coord system of camera 0
    # (each one in the SAME coord system) There will be Ncameras-1 of these
    points3d_at       = []


    for icam0 in xrange(0,Ncameras-1):
        # compute the estimated transform from camera icam0 to camera icam0+1 I
        # want the transform FROM the ref coord system (cam0) TO this camera
        rt10,IDs,points3d_here = estimate_consecutive_extrinsics(icam0)

        # points3d_here are 3d points common to views icam0,icam0+1 is in the
        # coord system of icam0

        # anything behind the camera, flip it to the front
        index = points3d_here[:,2] < 0
        points3d_here[index,:] *= -1

        # anything with a crazy-high range, scale it down
        range_here = np.sqrt(nps.inner(points3d_here,points3d_here))
        range_mean = np.mean(range_here)
        index = range_here > range_max
        if np.any(index):
            points3d_here[index,:] *= range_max/nps.transpose(range_here[index])

        # # scale all the ranges to a common mean
        # range_mean = np.mean(np.sqrt(nps.inner(points3d_here,points3d_here)))
        # points3d_here *= range_nominal/range_mean

        points3d_at_local.append(points3d_here)


        if icam0 == 0:
            # cam0 IS the ref coord system. Just store it
            extrinsics = nps.atleast_dims(rt10, -2)

        else:
            points3d_here = \
                mrcal.poseutils.transform_point_rt(mrcal.poseutils.invert_rt(extrinsics[-1]),
                                                   points3d_here)

            extrinsics = nps.glue(extrinsics,
                                  mrcal.poseutils.compose_rt(rt10, extrinsics[-1]),
                                  axis=-2)

        # I overwrite past point estimates here. Makes coding simpler
        points3d[ [ID_to_index[i] for i in IDs],: ] = points3d_here
        points3d_at.append(points3d_here)


    # args = [ ( points3d_at[i][:,0],
    #            points3d_at[i][:,1],
    #            points3d_at[i][:,2],
    #            np.sqrt(nps.inner(points3d_at[i],points3d_at[i])),
    #            dict(tuplesize=4,
    #                 _with='points palette pt 7 ps 2',
    #                 legend=str(i))) for i in xrange(len(points3d_at))]
    # gp.plot3d(*args, _with='points')

    # p0 = np.loadtxt(track_filename(0))[:,1:]
    # p1 = np.loadtxt(track_filename(1))[:,1:]
    # e10 = extrinsics[0]
    # e20 = extrinsics[1]
    # def rms(d):
    #     return np.sqrt(np.mean(nps.inner(d,d)))
    # print rms(mrcal.projections.project(points3d_at[0],intrinsics) - p0[:,:2])
    # print rms(mrcal.projections.project(mrcal.poseutils.transform_point_rt(e10, points3d_at[0]),intrinsics) - p0[:,2:])
    # print rms(mrcal.projections.project(points3d_at_local[1],intrinsics) - p1[:,:2])
    # print rms(mrcal.projections.project(mrcal.poseutils.transform_point_rt(e10, points3d_at[1]),intrinsics) - p1[:,:2])
    # print rms(mrcal.projections.project(mrcal.poseutils.transform_point_rt(e20, points3d_at[1]),intrinsics) - p1[:,2:])

    # plot = utils.visualize_solution(intrinsics_data_all, extrinsics,
    #                                 None, points3d,
    #                                 None, None,
    #                                 observations_point, indices_point_camera_points,
    #                                 distortion_model,
    #                                 axis_scale=100.0)

    # import IPython
    # IPython.embed()
    # sys.exit()


    return extrinsics,points3d




# gp.plot(observations_point[...,0], observations_point[...,1])

# Each "frame" is an instant in time. As far as mrcal is concerned, each is a
# different "camera".

distortion_model = 'LENSMODEL_PINHOLE'
intrinsics_data_one = np.array((focallength,focallength, imagersize[0]/2, imagersize[1]/2), dtype=float)
intrinsics_data_all = np.array(list(intrinsics_data_one)*Nframes, dtype=float).reshape(Nframes, 4)



# extrinsics = np.random.rand(Nframes-1, 6)* 1e-3
# points3d   = np.random.rand(Nfeatures, 3)* 1.0
# extrinsics[:,3:] = np.array((135.01428289,-150.85947397,-53.01273416))*nps.transpose(np.arange(len(extrinsics)) + 1)
# points3d[:,2]  += range0

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
observations_point[0,2] = range0 # I have ONE reference range


extrinsics0 = extrinsics.copy()
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

# p = mrcal.project(points3d, distortion_model, intrinsics_data_one)
# gp.plot(p[:,0],p[:,1], points3d[:,2], rgbimage='000.jpg', _with='points pt 7 ps 1 palette',
#         square=1, tuplesize=3, set=['xrange [*:*] noextend', 'yrange [*:*] reverse noextend'],
#         title='UAVSAR preliminary tracking results. {} tracked features over {} frames'.format(Nfeatures,Nframes),
#         cbrange=[10000,20000],
#         #hardcopy='UAVSAR_preliminary_tracking_results_{}features_{}frames.pdf'.format(Nfeatures,Nframes)
# )

# points3d_zeromean = points3d - np.mean(points3d, axis=-2)
# l,v = np.linalg.eig(nps.matmult(nps.transpose(points3d_zeromean),points3d_zeromean))
# gp.plot(*nps.transpose(nps.matmult(points3d_zeromean,v)), _with='points palette pt 7 ps 2', tuplesize=3)

import IPython
IPython.embed()







# These define the local cartesian coordinate system. Will be filled-in the
# first time we use these
_proj = None
def ne_from_latlon(lat,lon):
    '''Convenience function to interface to proj

    This is very helpful for avoiding bugs. Note that I want the opposite order
    for both the inputs and the outputs from what proj provides. Assumes _proj
    is already set up

    '''

    global _proj

    if _proj is None:
        # UTM coord system not initialized. Set it up, and save it for later
        from pyproj import Proj

        zone = 1 + int((lon + 180) // 6)

        _proj = \
            Proj(proj           = 'utm',
                 zone           = zone,
                 ellps          = 'WGS84',
                 preserve_units = False # output in meters
                 )

    e,n = _proj(lon, lat)
    return np.array((n,e))
