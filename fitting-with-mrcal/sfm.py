#!/usr/bin/python3

import sys
import numpy as np
import numpysane as nps
import re
import glob
import exifread
from pyproj import Proj
import vnlog
import cv2
import img_any

# Reference coordinate system: UTM, at altitude=0


sys.path[:0] = '/home/dima/src_boats/mrcal2/',

import mrcal


range_nominal = 12000 # more than altitude, because this is a skewed view
focal_length  = 4000

i_image_range = (8,71)


def get_image_paths(filename_hugin, imagedir):
    images = []

    with open(filename_hugin) as f:
        for l in f:
            m  = re.match(f"i .*\"(.*?)\"$", l)
            if m is not None:
                images.append(imagedir + '/' + m.group(1))


    if i_image_range is not None: return images[i_image_range[0]:i_image_range[1]]


    return images


def parse_hugin_features(filename_hugin):
    re_pos_int = '([0-9]+)'
    re_float   = '(-?[0-9\.-]+)'

    pointset_map  = dict()
    Nobservations = 0

    with open(filename_hugin) as f:
        for l in f:
            m  = re.match(f"c n{re_pos_int} N{re_pos_int} x{re_float} y{re_float} X{re_float} Y{re_float} t0", l)
            if m is not None:

                iframe0 = int(  m.group(1))
                x0      = float(m.group(3))
                y0      = float(m.group(4))
                iframe1 = int(  m.group(2))
                x1      = float(m.group(5))
                y1      = float(m.group(6))

                if i_image_range is not None:
                    if iframe0 <  i_image_range[0] or \
                       iframe1 <  i_image_range[0] or \
                       iframe0 >= i_image_range[1] or \
                       iframe1 >= i_image_range[1]:
                        continue
                    iframe0 -= i_image_range[0]
                    iframe1 -= i_image_range[0]

                key0 = '_'.join((str(iframe0), m.group(3), m.group(4)))
                key1 = '_'.join((str(iframe1), m.group(5), m.group(6)))

                pointset0 = pointset_map.get(key0)
                pointset1 = pointset_map.get(key1)

                if pointset0 is None and pointset1 is None:
                    # both sides are observing a new point
                    pointset = dict( _set       = set((key0, key1)),
                                     iframe_x_y = np.array(((iframe0, x0, y0),
                                                            (iframe1, x1, y1)), dtype=float))
                    pointset_map[key0] = pointset
                    pointset_map[key1] = pointset
                    Nobservations += 2
                elif pointset0 is None and pointset1 is not None:
                    pointset1['_set'].add(key0)
                    pointset1['iframe_x_y'] = \
                        nps.glue(pointset1['iframe_x_y'],
                                 np.array((iframe0, x0, y0), dtype=float),
                                 axis = -2)
                    pointset_map[key0] = pointset1
                    Nobservations += 1
                elif pointset0 is not None and pointset1 is None:
                    pointset0['_set'].add(key1)
                    pointset0['iframe_x_y'] = \
                        nps.glue(pointset0['iframe_x_y'],
                                 np.array((iframe1, x1, y1), dtype=float),
                                 axis = -2)
                    pointset_map[key1] = pointset0
                    Nobservations += 1
                else:
                    # both sides are looking at a known point. Need to merge pointset0 and pointset1
                    if pointset0 is not pointset1:
                        pointset0['_set'].update(pointset1['_set'])
                        pointset0['iframe_x_y'] = \
                            nps.glue(pointset0['iframe_x_y'],
                                     pointset1['iframe_x_y'],
                                     axis = -2)
                        for k in pointset1['_set']:
                            pointset_map[k] = pointset0

    iobservation        = 0
    observations        = np.zeros((Nobservations,2), dtype=float)
    indices_point_frame = np.zeros((Nobservations,2), dtype=np.int32)

    observations_done = set()
    ipoint = 0
    for k in pointset_map:
        if k in observations_done:
            continue

        iframe_x_y = pointset_map[k]['iframe_x_y']
        iordered   = np.argsort(iframe_x_y[:,0])

        # I can get a duplicate if two different points in a single image
        # transitively map to the same world point (if hugin says there's a
        # single correspondence in another image that maps to both of these). I
        # don't want to deal with this, so I ignore the whole point set
        if np.min(np.diff(iframe_x_y[iordered,0])) == 0:
            print(f"Nonunique point {ipoint}. Throwing away. iframe_x_y =\n{iframe_x_y}. ")
            observations_done.update(pointset_map[k]['_set'])
            Nobservations -= len(iframe_x_y)
            continue

        for i in iordered:
            observations[iobservation,:] = iframe_x_y[i,1:]
            indices_point_frame[iobservation,:] = \
                np.array( [ipoint, iframe_x_y[i,0]], dtype=np.int32)
            iobservation += 1

        observations_done.update(pointset_map[k]['_set'])
        ipoint += 1

    observations        = observations       [:Nobservations]
    indices_point_frame = indices_point_frame[:Nobservations]

    if Nobservations != iobservation:
        raise Exception("Nobservations mismatch. This is a data parsing bug")

    return indices_point_frame, observations, ipoint


def find_homography(i0, i1, indices_point_frame, observations):

    # This is an inefficient linear search, but I want to say that I have
    # little-enough data that this doesn't matter.

    # I find all points observed in images i0 and i1
    i0_mask = indices_point_frame[:,1] == i0
    i1_mask = indices_point_frame[:,1] == i1
    common, observation_indices0, observation_indices1 = \
        np.intersect1d( indices_point_frame[ i0_mask, 0 ],
                        indices_point_frame[ i1_mask, 0 ],
                        assume_unique  = True,
                        return_indices = True)

    q0 = observations[i0_mask.nonzero()[0][observation_indices0], :]
    q1 = observations[i1_mask.nonzero()[0][observation_indices1], :]
    s = cv2.findHomography(q0,q1, method=cv2.RANSAC, ransacReprojThreshold=30)
    return s[0]


def parse_known_position_points(gcp_vnl,
                                images,
                                W,H,
                                indices_point_frame, observations):

    points_fixed              = np.zeros((0,3), dtype=float)
    observations_fixed        = np.zeros((0,2), dtype=float)
    indices_point_frame_fixed = np.zeros((0,2), dtype=np.int32)
    point_labels              = []

    ipoint = 0
    for d in vnlog.vnlog(gcp_vnl):


        i0 = int (d['img0'])
        if i_image_range is not None:
            if i0 < i_image_range[0] or i0 >= i_image_range[1]:
                continue
            i0 -= i_image_range[0]

        # I grab the known gcp coordinates
        name = d['name']
        E    = float(d['E'])
        N    = float(d['N'])
        altm = float(d['altm'])
        points_fixed = nps.glue(points_fixed,
                                np.array((N,E,-altm), dtype=float),
                                axis=-2)
        point_labels += [name]

        # I use the feature-matcher to find the point observations of the ground
        # control points
        x0 = float(d['x0'  ])
        y0 = float(d['y0'  ])
        q0 = np.array((x0,y0), dtype=float)

        image0 = cv2.imread(images[i0], cv2.IMREAD_GRAYSCALE)
        found_any_matches_for_this_gcp = False
        for dframe in range(-4,5):
            i1 = i0 + dframe
            if i1 == i0:                    continue
            if i1 < 0 or i1 >= len(images): continue

            H10 = find_homography(i0, i1, indices_point_frame, observations)
            if H10 is None:
                continue

            q1 = mrcal.apply_homography(H10, q0)
            if q1[0] <= 0   or q1[1] <= 0 or \
               q1[0] >= W-1 or q1[1] >= H-1:
                continue

            image1 = cv2.imread(images[i1], cv2.IMREAD_GRAYSCALE)

            try:
                match = \
                  img_any.warp_template_and_correlate( \
                    image0, image1,
                    (31,31),
                    'XCORR_NORMALIZED',
                    search_radius  = 10,
                    corr_floor     = 0.9,
                    feat2_in       = q1,
                    corr_count     = 1,
                    H12            = np.linalg.inv(H10).astype(np.float32),
                    debug_filename = None
                    )
            except Exception as e:
                continue

            if not found_any_matches_for_this_gcp:
                observations_fixed = nps.glue(observations_fixed, q0, axis = -2)
                indices_point_frame_fixed = \
                    nps.glue(indices_point_frame_fixed,
                             np.array((ipoint, i0), dtype=np.int32),
                             axis = -2)

            q1 = np.array(match['feat2_out'], dtype=float)
            observations_fixed = nps.glue(observations_fixed, q1, axis = -2)
            indices_point_frame_fixed = \
                nps.glue(indices_point_frame_fixed,
                         np.array((ipoint, i1), dtype=np.int32),
                         axis = -2)
            found_any_matches_for_this_gcp = True

        if not found_any_matches_for_this_gcp:
            raise Exception(f"No matches for GCP '{name}' found!")

        ipoint += 1

    return indices_point_frame_fixed, observations_fixed, points_fixed, point_labels


def get_imagersize(image):
    H,W = cv2.imread(image).shape[:2]
    return \
        np.array((W,H), dtype=np.int32)

# These define the local cartesian coordinate system. Will be filled-in the
# first time we use these
_proj = None
def ne_from_latlon(lat,lon):
    '''Convenience function to interface to proj

    Note that proj takes lat,long and reports E,N. I flip the order of both the
    inputs and outputs for MY convenience
    '''

    global _proj

    if _proj is None:
        # UTM coord system not initialized. Set it up, and save it for later
        zone = 1 + int((lon + 180) // 6)

        _proj = \
            Proj(proj           = 'utm',
                 zone           = zone,
                 ellps          = 'WGS84',
                 preserve_units = False # output in meters
                 )

    e,n = _proj(lon, lat)
    return np.array((n,e))


def get_extrinsics_fromref_seed(images,
                                view_bearing_abs_deg,
                                view_pitch_below_horizontal_deg):
    r'''Compute the extrinsic seed

    The images are geotagged with the position. I assume a linear motion, and
    interpolate the translation by looking at the first and last lat/lon/alt
    points. I assume each view has the same orientation, with the yaw, pitch
    given in the argument

    '''

    ########### orientation
    # bearing is from N towards E

    # main coordinate system: UTM zone 11 at altitude=0
    # xyz: (N,E,down). SO THE HIGHER WE GO, THE MORE NEGATIVE THE Z
    #
    # camera coord system: xyz: (right,down,forward)
    cy = np.cos(np.pi/180.0 * view_bearing_abs_deg)
    sy = np.sin(np.pi/180.0 * view_bearing_abs_deg)

    cp = np.cos(np.pi/180.0 * view_pitch_below_horizontal_deg)
    sp = np.sin(np.pi/180.0 * view_pitch_below_horizontal_deg)

    v_ref_camforward = np.array(( cp*cy,  cp*sy, sp))
    v_ref_camright   = np.array((   -sy,     cy,  0))
    v_ref_camdown    = np.array((-sp*cy, -sp*sy, cp))

    R_cam_ref = nps.cat( v_ref_camright,
                         v_ref_camdown,
                         v_ref_camforward)


    ########### translation

    # This is ridiculously complicated, and there's no documentation. Not
    # entirely clear I'm doing this right
    def lat_lon_altm(filename):
        with open(filename, 'rb') as f:
            tags = exifread.process_file(f)
            lat,lon = exifread.utils.get_gps_coords(tags)
            alt = eval( tags['GPS GPSAltitudeRef'].printable + '+' + tags['GPS GPSAltitude'].printable)
        return lat,lon,alt

    lat0,lon0,altm0 = lat_lon_altm(images[ 0])
    ne0 = ne_from_latlon(lat0,lon0)

    lat1,lon1,altm1 = lat_lon_altm(images[-1])
    ne1 = ne_from_latlon(lat1,lon1)

    N = len(images)
    xyz0 = np.array((*ne0, -altm0), dtype=float)
    xyz1 = np.array((*ne1, -altm1), dtype=float)
    xyz = xyz0 + nps.transpose(np.arange(N, dtype=float))/(N-1) * (xyz1-xyz0)

    rt_ref_cam = np.zeros((N,6), dtype=float)
    rt_ref_cam[:,3:] = xyz
    rt_ref_cam[:,:3] = mrcal.r_from_R(nps.transpose(R_cam_ref))

    return mrcal.invert_rt(rt_ref_cam)


def get_points_seed(intrinsics, Npoints,
                    rt_camera_ref, range_nominal,
                    indices_point_frame, observations):

    # I find the first observation of each point, and unproject that ray some
    # nominal distance back. I ignore all the other observations, but that's ok
    # for the seed

    points = np.zeros((Npoints, 3), dtype=float)

    iobservations_sorted = np.argsort(indices_point_frame[:,0])

    isorted = np.searchsorted( indices_point_frame[iobservations_sorted][:,0],
                               np.arange(len(points), dtype=np.int32) )
    for ipoint in range(len(isorted)):
        # iobservation is an observation for point ipoint
        iobservation = iobservations_sorted[isorted[ipoint]]

        vcam = mrcal.unproject(observations[iobservation], *intrinsics)
        pcam = vcam / nps.mag(vcam) * range_nominal
        points[ipoint,:] = \
            mrcal.transform_point_rt( mrcal.invert_rt(rt_camera_ref[indices_point_frame[iobservation,1],:]),
                                      pcam )
    return points


imagedir       = 'images-downsampled-and-equalized'
hugin_filename = 'images-downsampled-and-equalized/pano.pto'

images = get_image_paths(hugin_filename, imagedir)

lensmodel  = 'LENSMODEL_PINHOLE'
imagersize = get_imagersize( images[0] )
intrinsics_data = np.array((focal_length, focal_length, (imagersize[0]-1.)/2., (imagersize[1]-1.)/2.), dtype=float)

indices_point_frame, observations, Npoints = \
    parse_hugin_features(hugin_filename)

if not( np.min(indices_point_frame[:,1]) == 0 and
        np.max(indices_point_frame[:,1]) == len(images)-1 ):
    raise Exception(f"Expected hugin images from 0 to {len(images)-1}, but got from {np.min(indices_point_frame[:,1])} to {np.max(indices_point_frame[:,1])}")


indices_point_frame_fixed, observations_fixed, points_fixed, point_labels = \
    parse_known_position_points("gcp.vnl",
                                images,
                                imagersize[0], imagersize[1],
                                indices_point_frame, observations)
print(f"Got {len(points_fixed)} fixed points")

extrinsics_rt_fromref = get_extrinsics_fromref_seed(images,
                                                    240., # looking SW
                                                    40    # pitched down somewhat
                                                    )
points = get_points_seed( (lensmodel,intrinsics_data), Npoints,
                          extrinsics_rt_fromref, range_nominal,
                          indices_point_frame, observations )

# join the floating and fixed point data. The fixed points lie at the end of the
# list, after the floating points
points = nps.glue(points, points_fixed, axis=-2)
indices_point_frame_fixed[:,0] += Npoints
indices_point_frame = nps.glue(indices_point_frame,
                               indices_point_frame_fixed,
                               axis=-2)
observations = nps.glue(observations,
                        observations_fixed,
                        axis=-2)

indices_point_camintrinsics_camextrinsics_flags = \
    nps.glue( indices_point_frame[:, (0,)],
              np.zeros( (len(indices_point_frame), 1), dtype=np.int32),
              indices_point_frame[:, (1,)],
              np.zeros( (len(indices_point_frame), 1), dtype=np.int32),
              axis = -1 )

# Add a unit weight column
observations = nps.glue(observations,
                        np.ones((len(observations), 1), dtype=float),
                        axis = -1)


if 0:

    x,J = mrcal.optimizerCallback( nps.atleast_dims(intrinsics_data, -2),
                                   extrinsics_rt_fromref,
                                   None, points,
                                   None, None,
                                   observations, indices_point_camintrinsics_camextrinsics_flags,
                                   lensmodel,
                                   Npoints_fixed                     = len(points_fixed),
                                   imagersizes                       = nps.atleast_dims(imagersize, -2),

                                   do_optimize_intrinsic_core        = False,
                                   do_optimize_intrinsic_distortions = False,

                                   do_optimize_extrinsics            = True,
                                   do_optimize_frames                = True,

                                   skip_regularization               = False,

                                   verbose                           = True)

else:

    stats = mrcal.optimize( nps.atleast_dims(intrinsics_data, -2),
                            extrinsics_rt_fromref,
                            None, points,
                            None, None,
                            observations, indices_point_camintrinsics_camextrinsics_flags,
                            lensmodel,
                            Npoints_fixed                     = len(points_fixed),
                            imagersizes                       = nps.atleast_dims(imagersize, -2),
                            observed_pixel_uncertainty        = 1.0,
                            do_optimize_intrinsic_distortions = False,

                            do_optimize_intrinsic_core        = False,
                            do_optimize_extrinsics            = True,
                            do_optimize_frames                = True,

                            skip_outlier_rejection            = True,
                            skip_regularization               = False,
                            verbose                           = True)



print(stats['rms_reproj_error__pixels'])

x = stats['x']
point_labels_map = dict()
for i_point_fixed in range(len(points_fixed)):
    point_labels_map[i_point_fixed + len(points)-len(points_fixed)] = \
        point_labels[i_point_fixed]
mrcal.show_calibration_geometry( extrinsics_rt_fromref,
                                 points=points,
                                 axis_scale=1000,
                                 point_labels = point_labels_map,
                                 hardcopy='/tmp/tst.gp' )
