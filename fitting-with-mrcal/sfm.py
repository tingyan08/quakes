#!/usr/bin/python3

r'''Compute the geometry of the world using observations from moving cameras

SYNOPSIS

  $ blah

  woah

This tool solves structure-from-motion problems

'''

import sys
import argparse
import re
import os

def parse_args():

    parser = \
        argparse.ArgumentParser(description = __doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--focal',
                        type=float,
                        required = True,
                        help='Initial estimate of the focal length, in pixels')
    parser.add_argument('--explore',
                        action='store_true',
                        required=False,
                        default=False,
                        help='''After the solve open an interactive shell to examine the solution''')

    return parser.parse_args()

args = parse_args()

# arg-parsing is done before the imports so that --help works without building
# stuff, so that I can generate the manpages and README






import numpy as np
import numpysane as nps
import glob
import exifread
from pyproj import Proj
import vnlog
import cv2
import img_any
import copy

# Reference coordinate system: UTM, at altitude=0


sys.path[:0] = '/home/dima/src_boats/mrcal2/',

import mrcal
import deltapose_lite


# sfm tool for mrcal
# sfm tool should have visualization; show-residuals should be shared
# need nice uncertainty quantification, including intrinsics maps
# better automatic seeding? Detection of convergence failure?
# outlier rejection for points

# seed with deltapose. This would get the cam poses AND the points
# evaluate plane altitude
# evaluate the GCP error
# evaluate the ground height error


range_nominal  = 16000
imagedir       = 'images-downsampled-and-equalized'
hugin_filename = 'images-downsampled-and-equalized/pano.frameno414-pushed-back-5.pto'
gcp_filename   = 'gcp.vnl'
image_range  = (40,42) # two frames; show fixed point seed to a very high cost
image_range  = (22,63)
image_range  = (0,85)
image_range  = [33,43]
image_range  = [53,70]

#image_range  = [23,68]
image_range  = [33,68]
image_range  = [32,79]

# used to get the optimal f
image_range  = [32,75]

image_range  = [32,82]



camera_position_bias = np.array([-3428.83582502,2772.04308145,289.3221032 ])





def fill_with_empty_dict_until_i(l, i):
    r'''Adds dict() elements to list l to make sure l[i] exists'''

    for _ in range(len(l),i+1):
        l.append(dict())


def get_image_paths(filename_hugin, imagedir):
    images = []

    with open(filename_hugin) as f:
        for l in f:
            m  = re.match(f"i .*\"(.*?)\"$", l)
            if m is not None:
                images.append(imagedir + '/' + m.group(1))

    return images


def parse_hugin_features(filename_hugin):
    r'''Parses a hugin .pto file into a working form

    Returns

    - frames[] list, indexable on frame index. Each element is a dict with point
      indices as keys, observation indices as values. Used like this:
      observations[ frames[i_frame][i_point] ]

    - points[] list, indexable on point index. Each element is a dict with frame
      indices as keys, observation indices as values. Used like this:
      observation[ points[i_point][i_frame] ]

    - observations[] numpy array: the pixel coordinates. frames[] and points[]
      reference this

    This is clearly a redundant way to specify this, but it simplifies
    downstream usage

    '''
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

    iobservation = 0
    observations = np.zeros((Nobservations,2), dtype=float)

    points = []
    frames = []


    observations_done = set()
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

        ipoint = len(points)
        points.append(dict())

        for i in iordered:
            iframe = int(iframe_x_y[i,0])
            fill_with_empty_dict_until_i(frames, iframe)

            points[ipoint][iframe] = iobservation
            frames[iframe][ipoint] = iobservation

            observations[iobservation,:] = iframe_x_y[i,1:]
            iobservation += 1

        observations_done.update(pointset_map[k]['_set'])


    # we may have ignored some observations, so I cut off the tail of the list
    # that never got filled in
    observations = observations[:Nobservations]

    if Nobservations != iobservation:
        raise Exception("Nobservations mismatch. This is a data parsing bug")

    return frames,points,observations


def find_homography(iframe0, iframe1, frames, observations):

    ipoint_common = sorted( frames[iframe0].keys() & frames[iframe1].keys() )
    q0 = np.zeros((len(ipoint_common),2), dtype=float)
    q1 = np.zeros((len(ipoint_common),2), dtype=float)
    for i in range(len(ipoint_common)):
        q0[i,:] = observations[frames[iframe0][ipoint_common[i]]]
        q1[i,:] = observations[frames[iframe1][ipoint_common[i]]]

    s = cv2.findHomography(q0,q1, method=cv2.RANSAC, ransacReprojThreshold=30)
    return s[0]


def parse_fixed_points(gcp_vnl,
                       images,
                       W,H,
                       frames_variable, points_variable, observations_variable):

    r'''Parses the known-location points

    The outputs have the same structure as those in parse_hugin_features(), with
    an extra output:

    - point_positions: a numpy array of shape (Npoints,3) containing the
      reference-coordinate positions of the observed points

    '''

    points          = []
    frames          = []
    point_labels    = []
    point_positions = np.zeros((0,3), dtype=float)
    observations    = np.zeros((0,2), dtype=float)

    ipoint = 0
    for d in vnlog.vnlog(gcp_vnl):

        fill_with_empty_dict_until_i(points, ipoint)

        i0   = int (d['img0'])
        name = d['name']

        # I grab the known gcp coordinates
        E    = float(d['E'])
        N    = float(d['N'])
        altm = float(d['altm'])
        point_positions = nps.glue(point_positions,
                                   np.array((N,E,-altm), dtype=float),
                                   axis=-2)
        point_labels += [name]

        # I use the feature-matcher to find the point observations of the ground
        # control points
        q0 = np.array((float(d['x0']), float(d['y0'])), dtype=float)

        image0 = cv2.imread(images[i0], cv2.IMREAD_GRAYSCALE)
        found_any_matches_for_this_gcp = False

        # I look for matches in a neighborhood of frames around the one I know
        # about
        for dframe in range(-4,5):
            i1 = i0 + dframe
            if i1 == i0:                    continue
            if i1 < 0 or i1 >= len(images): continue

            H10 = find_homography(i0, i1, frames_variable, observations_variable)
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
                    (15,15),
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
                print(f"Adding fixed point {name} defined at frame_index {i0}")
                iframe       = i0
                iobservation = len(observations)
                fill_with_empty_dict_until_i(frames, iframe)
                points[ipoint][iframe] = iobservation
                frames[iframe][ipoint] = iobservation
                observations = nps.glue(observations, q0, axis = -2)

            q1 = np.array(match['feat2_out'], dtype=float)

            iframe       = i1
            iobservation = len(observations)
            fill_with_empty_dict_until_i(frames, iframe)
            points[ipoint][iframe] = iobservation
            frames[iframe][ipoint] = iobservation
            observations = nps.glue(observations, q1, axis = -2)

            found_any_matches_for_this_gcp = True

        if not found_any_matches_for_this_gcp:
            # I already set point_positions and point_labels
            raise Exception(f"No matches for GCP '{name}' found!")

        ipoint += 1

    return frames, points, observations, point_positions, point_labels


def merge_variable_fixed( frames_variable, points_variable, observations_variable,
                          frames_fixed,    points_fixed,    observations_fixed ):
    r'''Merges the variable-point and fixed-point data

    These are output from parse_hugin_features() and parse_fixed_points()
    respectively. The structure of the inputs is described in
    parse_hugin_features(). Note that the two input sets use the same domain to
    index frames, but a separate domain to index points and observations

    '''

    Nobservations_variable = len(observations_variable)
    Npoints_variable       = len(points_variable)

    observations = nps.glue(observations_variable, observations_fixed,
                            axis = -2)

    frames = copy.deepcopy(frames_variable)
    for iframe in range(len(frames_fixed)):
        for ipoint_fixed in frames_fixed[iframe].keys():
            frames[iframe][ipoint_fixed + Npoints_variable] = \
                frames_fixed[iframe][ipoint_fixed] + Nobservations_variable

    points = copy.deepcopy(points_variable)
    fill_with_empty_dict_until_i(points, len(points_variable) + len(points_fixed) - 1)
    for ipoint_fixed in range(len(points_fixed)):
        for iframe in points_fixed[ipoint_fixed].keys():
            points[ipoint_fixed + Npoints_variable][iframe] = \
                points_fixed[ipoint_fixed][iframe] + Nobservations_variable

    return frames,points,observations


def get_frame_subset( frames, points, observations,
                      point_fixed_positions, Npoints_variable,
                      subset ):
    r'''Filters the input to keep ONLY frames from a given subset

    The structure of the input and output is described in
    parse_hugin_features(). Any points with less than 2 observations are thrown
    out. All the indices are modified to stay sequential, starting at 0

    '''

    point_set = set()
    for frame in frames:
        point_set |= frame.keys()

    ipoint_sorted = sorted(point_set)

    points_subset = []
    for ipoint in ipoint_sorted:
        point = points[ipoint]
        point_here = dict()
        for iframe in sorted(point.keys()):
            if iframe < subset[0] or iframe >= subset[1]: continue
            point_here[iframe - subset[0]] = point[iframe]
        if len(point_here.keys()) < 2:
            # points with a single observation aren't useful, so I throw them
            # out
            point_set.remove(ipoint)
            continue
        points_subset.append(point_here)

    # some points were removed, so I recompute the list
    ipoint_sorted = sorted(point_set)

    # map from the full ipoint to the subset ipoint
    ipoint_map = dict()
    for i in range(len(ipoint_sorted)):
        ipoint_map[ipoint_sorted[i]] = i

    frames_subset = []
    for frame in frames[slice(*subset)]:
        frame_here = dict()
        for ipoint in frame.keys():
            if ipoint in point_set:
                frame_here[ipoint_map[ipoint]] = frame[ipoint]
        frames_subset.append(frame_here)

    # frames and points and now set. Now I need to compute the observations
    # subset, and apply the cut-down indices to the

    iobservation_set = set([iobservation for frame in frames_subset for iobservation in frame.values()])
    # map from the full iobservation to the subset iobservation
    iobservation_map = dict()
    iobservation_sorted = sorted(iobservation_set)
    for i in range(len(iobservation_sorted)):
        iobservation_map[iobservation_sorted[i]] = i

    for frame in frames_subset:
        for ipoint in frame.keys():
            frame[ipoint] = iobservation_map[frame[ipoint]]
    for point in points_subset:
        for iframe in point.keys():
            point[iframe] = iobservation_map[point[iframe]]

    observations_subset = np.zeros((len(iobservation_set), 2), dtype=float)
    for iobservation in range(len(observations)):
        if iobservation in iobservation_set:
            observations_subset[ iobservation_map[iobservation] ] = observations[iobservation]



    points_fixed_positions_subset = np.zeros((0,3), dtype=float)
    Nobservations_fixed_subset = 0
    for ipoint in sorted(ipoint_map.keys()):
        if ipoint >= Npoints_variable:
            points_fixed_positions_subset = \
                nps.glue( points_fixed_positions_subset,
                          point_fixed_positions[ipoint-Npoints_variable],
                          axis = -2)
            Nobservations_fixed_subset += len(points_subset[ipoint_map[ipoint]].keys())


    return frames_subset,points_subset,observations_subset, \
        ipoint_map, iobservation_map, \
        points_fixed_positions_subset, Nobservations_fixed_subset


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


def make_geometry_seed(images,
                       intrinsics,

                       # These are the floating AND fixed points
                       frames,points,observations,

                       # Just the fixed points. This function computes the others
                       points_fixed_positions,

                       # To make the initial point estimates reasonable
                       range_nominal,

                       # pre-computed state. These are a python list. Each
                       # element is either a numpy array of data to use or None,
                       # meaning we should estimate a seed
                       rt_cr_pre=None, points_pre=None ):
    r'''Compute the initial camera and point geometry estimates

    I make several assumptions that should get me close-enough:

    1. The images are geotagged; I want to use the geotagging for the estimate
       of the camera position, but the lat/lon has obvious aphysical jumps. So I
       take the first/last geotags, and linearly interpolate

    2. I assume that stuff we're looking at is much further out than the
       distance between consecutive camera positions, so I can use a procrustes
       fit to compute the rotation between consecutive camera positions. Now I
       have the absolute position of all the cameras, and a relative orientation
       of all the cameras. There's a single, unknown rotation that can be
       applied to all the cameras together

    3. I can compute the reference-coord point xr from a camera-i point xi:

         xr = Rri xi + tri

       I have an absolute translation, and relative rotation of my cameras, so I
       have tri, but Rri = R Rsi where I know Rsi, but need to find R. This R is
       the global orientation of the cameras.

         xi = Rsit Rt (xr - tri)

       I can compute R from the set of fixed points I have. Let's say camera i is
       observing fixed point pj at observation vector vij. What's the error?

         e(pj,i) = -inner( vij, Rsit Rt (pj - tri) / mag(pj - tri) )

       Here I have knowledge of everything except R, and this can be solved with
       a vanilla Procrustes fit

    4. I can then run a basic triangulation (Lindstrom, preferably; see the
       "Triangulation made easy" paper) to estimate the position of all the
       observed non-fixed points

    '''

    def get_geotagged_xyz(image_filename):
        r'''Return the position of the given image in my cartesian coordinates

        My coords are (northing,easting,down). I read the lat,lon,alt triplet
        from the EXIF, transform it to my coord system, and return

        '''
        def lat_lon_altm(filename):
            # This is ridiculously, and there's no documentation. I'm probably
            # not doing this right; the eval() is ridiculous, but it gets the
            # job done for this data
            with open(filename, 'rb') as f:
                tags = exifread.process_file(f)
                lat,lon = exifread.utils.get_gps_coords(tags)
                alt = eval( tags['GPS GPSAltitudeRef'].printable + '+' + tags['GPS GPSAltitude'].printable)
            return lat,lon,alt

        lat,lon,altm = lat_lon_altm(image_filename)
        ne = ne_from_latlon(lat,lon)
        return np.array((*ne, -altm), dtype=float) - camera_position_bias

    global camera_position_orig
    camera_position_orig = nps.cat( *[get_geotagged_xyz(i) for i in images]) + camera_position_bias



    Npoints_fixed    = len(points_fixed_positions)
    Npoints_all      = len(points)
    Npoints_variable = Npoints_all - Npoints_fixed

    if rt_cr_pre is not None:
        Rt_rc_pre = [mrcal.invert_Rt(mrcal.Rt_from_rt(rt)) if rt is not None else None for rt in rt_cr_pre]

    ####### 1. Get the absolute camera translation
    xyz0 = get_geotagged_xyz(images[0])
    xyz1 = get_geotagged_xyz(images[-1])
    N    = len(images)
    tri = nps.transpose(np.arange(N,dtype=float))/(N-1) * (xyz1-xyz0) + xyz0

    # use the precomputed values where I have them
    if rt_cr_pre is not None:
        for i in range(N):
            if Rt_rc_pre[i] is not None: tri[i] = Rt_rc_pre[i][3,:]


    ####### 2. Get the relative camera orientation
    # I compute the rotation from each camera's coord system to the system of
    # camera0. This is arbitrary. There will be some drift, since I'm
    # accumulating inaccurate measurements. This is hopefully OK
    Rsi    = np.zeros((len(images), 3, 3), dtype=float)
    Rsi[0] = np.eye(3)

    for i in range(1,len(images)):

        ipoint_common = sorted( frames[i-1].keys() & frames[i].keys() )
        q0 = np.zeros((len(ipoint_common),2), dtype=float)
        q1 = np.zeros((len(ipoint_common),2), dtype=float)
        for ipt in range(len(ipoint_common)):
            q0[ipt,:] = observations[frames[i-1][ipoint_common[ipt]]]
            q1[ipt,:] = observations[frames[i  ][ipoint_common[ipt]]]

        v0 = mrcal.unproject(q0, *intrinsics)
        v1 = mrcal.unproject(q1, *intrinsics)

        R01 = \
            mrcal.align3d_procrustes( v0 / nps.dummy(nps.mag(v0),-1),
                                      v1 / nps.dummy(nps.mag(v1),-1),
                                      vectors=True)







        Rextra = mrcal.R_from_r( 0.5*np.array([0, -nps.mag(tri[i]-tri[i-1])/range_nominal, 0]))
        R01 = nps.matmult(R01,Rextra)









        Rsi[i] = nps.matmult(Rsi[i-1], R01)


    ####### 3. Get the absolute camera orientation
    # From the derivation above:
    #   e(pj,i) = -inner( vij, Rsit Rt (pj - tri) / mag(pj - tri) )
    #           = - vijt Rsit Rt (pj - tri) / mag(pj - tri)
    #           = - (pj - tri)t / mag(pj - tri) R Rsi vij
    A = np.zeros((0, 3), dtype=float)
    B = np.zeros((0, 3), dtype=float)

    for j in range(Npoints_fixed):
        pj = points_fixed_positions[j]

        point_dict = points[-Npoints_fixed + j]
        for iframe in point_dict.keys():
            i_observation = point_dict[iframe]
            vij = mrcal.unproject(observations[i_observation], *intrinsics)

            A = nps.glue( A,
                          (pj - tri[iframe]) / nps.dummy(nps.mag(pj - tri[iframe]), -1),
                          axis = -2)
            B = nps.glue( B,
                          nps.inner(Rsi[iframe], vij),
                          axis = -2)

    R = mrcal.align3d_procrustes( A,B, vectors=True)
    Rri = nps.matmult(R, Rsi)

    # use the precomputed values where I have them
    if rt_cr_pre is not None:
        for i in range(N):
            if Rt_rc_pre[i] is not None: Rri[i] = Rt_rc_pre[i][:3,:]


    ####### 3.5. Store the camera pose
    Rt_ri = nps.glue( Rri,
                      nps.dummy(tri, -2),
                      axis = -2)
    Rt_ir = mrcal.invert_Rt(Rt_ri)
    extrinsics_rt_fromref = mrcal.rt_from_Rt( Rt_ir )


    ####### 4. Get the absolute point positions
    # I know where all the cameras are, triangulate.
    points_positions = np.zeros((Npoints_all, 3), dtype=float)

    # I do this one frame at a time, processing ALL the points. Each frame
    # overwrites the previous frame's results, which is fine. I also overwrite
    # the fixed point-positions; these are restored at the end of the loop
    def triangulate_points(i0, step):
        for i1 in range(i0+step, len(images), step):
            i0 = i1-step

            ipoint_common = sorted( frames[i0].keys() & frames[i1].keys() )
            if len(ipoint_common) == 0:
                continue

            q0 = np.zeros((len(ipoint_common),2), dtype=float)
            q1 = np.zeros((len(ipoint_common),2), dtype=float)
            for i in range(len(ipoint_common)):
                q0[i,:] = observations[frames[i0][ipoint_common[i]]]
                q1[i,:] = observations[frames[i1][ipoint_common[i]]]

            rt10 = mrcal.rt_from_Rt(mrcal.compose_Rt( Rt_ir[i1], Rt_ri[i0] ))

            p = deltapose_lite. \
                compute_3d_intersection_lindstrom(rt10, intrinsics, intrinsics,
                                                  q0, q1)

            p = mrcal.transform_point_Rt(Rt_ri[i0], p)

            # I have the points estimate now. Unfortunately, the geometry estimate
            # was sufficiently off to make this unreliable. I adjust the range to
            # more reasonable values where needed

            # midpoint between the two cameras, non-normalized mean "forward" vector
            p0 = (Rt_ri[i0, 3,:] + Rt_ri[i1, 3,:]) / 2.
            v0 = (Rt_ri[i0,:3,2] + Rt_ri[i1,:3,2]) / 2.

            # ranges I have
            r = np.copysign(nps.mag(p-p0), nps.inner(v0,p-p0))

            # which ranges need adjustment?
            i_point_adjust = (r < 0) + (r > range_nominal*2)

            if np.any(i_point_adjust):
                p[i_point_adjust,:] = (p[i_point_adjust,:]-p0) / nps.dummy(r[i_point_adjust],-1) * range_nominal + p0

            points_positions[ipoint_common,:] = p

    # I need a pairwise correspondence to seed each point. Somehow some points
    # only have a correspondence LOTS of frames away, so I look at all the
    # pairs. This is inefficient, but will do for now
    for step in range(1,13):
        for i0 in range(step):
            triangulate_points(i0, step)
        if not np.any(nps.norm2(points_positions) == 0):
            break
    else:
        raise Exception("Some points not seeded. No near-enough correspondences")



    # I overwrote the fixed points. Put them back
    points_positions[-Npoints_fixed:] = points_fixed_positions

    # use the precomputed values where I have them
    if points_pre is not None:
        for i in range(len(points_positions)):
            if points_pre[i] is not None: points_positions[i,:] = points_pre[i]


    return extrinsics_rt_fromref, points_positions


def make_input(frames, points, observations):
    '''Assumes the input data is dense, but does NOT assume sorted observations'''

    Nobservations = len(observations)
    observations_resorted = observations*0

    indices_point_frame = np.zeros((Nobservations, 2), dtype=np.int32)
    iobservation = 0
    for ipoint in range(len(points)):
        point = points[ipoint]
        for iframe in sorted(point.keys()):

            indices_point_frame  [iobservation,0] = ipoint
            indices_point_frame  [iobservation,1] = iframe
            observations_resorted[iobservation,:] = observations[ point[iframe],: ]
            iobservation += 1

    if Nobservations != iobservation:
        raise Exception("Nobservations mismatch. This is a data parsing bug")

    return indices_point_frame, observations_resorted







images = get_image_paths(hugin_filename, imagedir)

lensmodel  = 'LENSMODEL_PINHOLE'
imagersize = get_imagersize( images[0] )
intrinsics_data = np.array((args.focal, args.focal, (imagersize[0]-1.)/2., (imagersize[1]-1.)/2.), dtype=float)

frames_variable, points_variable, observations_variable = \
    parse_hugin_features(hugin_filename)

# These use the same set of frames for indices as the variable-point data, but
# the set of points and observations start counting at 0
frames_fixed, points_fixed, observations_fixed, point_fixed_positions, point_labels = \
    parse_fixed_points(gcp_filename,
                       images,
                       imagersize[0], imagersize[1],
                       frames_variable, points_variable, observations_variable)

frames,points,observations = \
    merge_variable_fixed( frames_variable, points_variable, observations_variable,
                          frames_fixed,    points_fixed,    observations_fixed )





# maps are from full-data indices to subset indices
frames_subset,points_subset,observations_subset, \
ipoint_map, iobservation_map, \
points_fixed_positions_subset, Nobservations_fixed_subset = \
    get_frame_subset( frames, points, observations,
                      point_fixed_positions, len(points_variable),
                      image_range)
images_subset = images[slice(*image_range)]


# loop:
#   get seed
#   get_input_from_subset
#   solve

#   add more frames


Npoints_fixed_subset = len(points_fixed_positions_subset)
print(f"Solving with {Npoints_fixed_subset} fixed points")


extrinsics_rt_fromref,points_subset_positions = \
    make_geometry_seed(images_subset,
                       (lensmodel,intrinsics_data),

                       # These are the floating AND fixed points
                       frames_subset,points_subset,observations_subset,

                       # Just the fixed points. This function computes the others
                       points_fixed_positions_subset,

                       # To make the initial point estimates reasonable
                       range_nominal)

indices_point_frame, observations_subset_resorted = \
    make_input(frames_subset, points_subset, observations_subset)

indices_point_camintrinsics_camextrinsics_flags = \
    nps.glue( indices_point_frame[:, (0,)],
              np.zeros( (len(indices_point_frame), 1), dtype=np.int32),
              indices_point_frame[:, (1,)],
              np.zeros( (len(indices_point_frame), 1), dtype=np.int32),
              axis = -1 )

# Add a weight column
observations_solver = \
    nps.glue(observations_subset_resorted,
             np.ones((len(observations_subset_resorted), 1), dtype=float),
             axis = -1)
# # The fixed-point costs weight far more. This makes the solve more stable
# observations_subset_resorted[-Nobservations_fixed_subset:, 2] *= 100.









Nextra_at_once = 2
Nsets = 0 #4
for iextra in range(Nsets):
    stats = mrcal.optimize( nps.atleast_dims(intrinsics_data, -2),
                            extrinsics_rt_fromref,
                            None, points_subset_positions,
                            None, None,
                            observations_solver, indices_point_camintrinsics_camextrinsics_flags,
                            lensmodel,
                            Npoints_fixed                     = Npoints_fixed_subset,
                            imagersizes                       = nps.atleast_dims(imagersize, -2),
                            observed_pixel_uncertainty        = 1.0,
                            do_optimize_intrinsic_distortions = False,

                            do_optimize_intrinsic_core        = False,
                            do_optimize_extrinsics            = True,
                            do_optimize_frames                = True,

                            skip_outlier_rejection            = True,
                            skip_regularization               = False,
                            verbose                           = True)

    ipoint_map0              = ipoint_map
    extrinsics_rt_fromref0   = extrinsics_rt_fromref.copy()
    points_subset_positions0 = points_subset_positions.copy()


    image_range[1] += Nextra_at_once

    # maps are from full-data indices to subset indices
    frames_subset,points_subset,observations_subset, \
    ipoint_map, iobservation_map, \
    points_fixed_positions_subset, Nobservations_fixed_subset = \
        get_frame_subset( frames, points, observations,
                          point_fixed_positions, len(points_variable),
                          image_range)
    images_subset = images[slice(*image_range)]

    Npoints_fixed_subset = len(points_fixed_positions_subset)
    print(f"Solving with {Npoints_fixed_subset} fixed points")


    rt_cr_pre = list(extrinsics_rt_fromref0) + [None]*Nextra_at_once
    points_pre = [None] * len(points_subset)
    for ipoint in ipoint_map0.keys():
        ipoint_subset0 = ipoint_map0[ipoint]
        ipoint_subset1 = ipoint_map [ipoint]
        points_pre[ipoint_subset1] = points_subset_positions0[ipoint_subset0]

    extrinsics_rt_fromref,points_subset_positions = \
        make_geometry_seed(images_subset,
                           (lensmodel,intrinsics_data),

                           # These are the floating AND fixed points
                           frames_subset,points_subset,observations_subset,

                           # Just the fixed points. This function computes the others
                           points_fixed_positions_subset,

                           # To make the initial point estimates reasonable
                           range_nominal,

                           rt_cr_pre  = rt_cr_pre,
                           points_pre = points_pre)


    indices_point_frame, observations_subset_resorted = \
        make_input(frames_subset, points_subset, observations_subset)

    indices_point_camintrinsics_camextrinsics_flags = \
        nps.glue( indices_point_frame[:, (0,)],
                  np.zeros( (len(indices_point_frame), 1), dtype=np.int32),
                  indices_point_frame[:, (1,)],
                  np.zeros( (len(indices_point_frame), 1), dtype=np.int32),
                  axis = -1 )

    # Add a weight column
    observations_solver = \
        nps.glue(observations_subset_resorted,
                 np.ones((len(observations_subset_resorted), 1), dtype=float),
                 axis = -1)
    # # The fixed-point costs weight far more. This makes the solve more stable
    # observations_subset_resorted[-Nobservations_fixed_subset:, 2] *= 100.


































if 0:

    x,J = mrcal.optimizerCallback( nps.atleast_dims(intrinsics_data, -2),
                                   extrinsics_rt_fromref,
                                   None, points_subset_positions,
                                   None, None,
                                   observations_solver, indices_point_camintrinsics_camextrinsics_flags,
                                   lensmodel,
                                   Npoints_fixed                     = Npoints_fixed_subset,
                                   imagersizes                       = nps.atleast_dims(imagersize, -2),

                                   do_optimize_intrinsic_core        = False,
                                   do_optimize_intrinsic_distortions = False,

                                   do_optimize_extrinsics            = True,
                                   do_optimize_frames                = True,

                                   skip_regularization               = False,

                                   verbose                           = True)

    # between the two cameras in the ref coord system
    p0 = (mrcal.invert_rt(extrinsics_rt_fromref[0])[3:] +
          mrcal.invert_rt(extrinsics_rt_fromref[1])[3:]) / 2.
    for ipt in range(len(points_subset_positions)):
        iobs0 = ((indices_point_frame[:,0]==ipt)*(indices_point_frame[:,1]==0)).nonzero()[0]
        if len(iobs0) == 0: continue
        iobs1 = ((indices_point_frame[:,0]==ipt)*(indices_point_frame[:,1]==1)).nonzero()[0]
        if len(iobs1) == 0: continue

        intrinsics = ('LENSMODEL_PINHOLE',intrinsics_data)
        pfit_cam = \
            deltapose_lite.\
            compute_3d_intersection_lindstrom(mrcal.compose_rt(extrinsics_rt_fromref[1],
                                                               mrcal.invert_rt(extrinsics_rt_fromref[0])),
                                              intrinsics, intrinsics,
                                              observations_subset_resorted[iobs0,:2], observations_subset_resorted[iobs1,:2])
        print(repr(pfit_cam))
        pfit = mrcal.transform_point_rt( mrcal.invert_rt(extrinsics_rt_fromref[0]),
                                         pfit_cam)

else:

    # stats = mrcal.optimize( nps.atleast_dims(intrinsics_data, -2),
    #                         extrinsics_rt_fromref,
    #                         None, points,
    #                         None, None,
    #                         observations, indices_point_camintrinsics_camextrinsics_flags,
    #                         lensmodel,
    #                         Npoints_fixed                     = len(points_fixed),
    #                         imagersizes                       = nps.atleast_dims(imagersize, -2),
    #                         observed_pixel_uncertainty        = 1.0,
    #                         do_optimize_intrinsic_distortions = False,

    #                         do_optimize_intrinsic_core        = False,
    #                         do_optimize_extrinsics            = False,
    #                         do_optimize_frames                = True,

    #                         skip_outlier_rejection            = True,
    #                         skip_regularization               = False,
    #                         verbose                           = True)

    # stats = mrcal.optimize( nps.atleast_dims(intrinsics_data, -2),
    #                         extrinsics_rt_fromref,
    #                         None, points,
    #                         None, None,
    #                         observations, indices_point_camintrinsics_camextrinsics_flags,
    #                         lensmodel,
    #                         Npoints_fixed                     = len(points_fixed),
    #                         imagersizes                       = nps.atleast_dims(imagersize, -2),
    #                         observed_pixel_uncertainty        = 1.0,
    #                         do_optimize_intrinsic_distortions = False,

    #                         do_optimize_intrinsic_core        = False,
    #                         do_optimize_extrinsics            = True,
    #                         do_optimize_frames                = False,

    #                         skip_outlier_rejection            = True,
    #                         skip_regularization               = False,
    #                         verbose                           = True)

    rt_rc_orig = mrcal.invert_rt(extrinsics_rt_fromref)

    # stats = mrcal.optimize( nps.atleast_dims(intrinsics_data, -2),
    #                         extrinsics_rt_fromref,
    #                         None, points,
    #                         None, None,
    #                         observations, indices_point_camintrinsics_camextrinsics_flags,
    #                         lensmodel,
    #                         Npoints_fixed                     = len(points_fixed),
    #                         imagersizes                       = nps.atleast_dims(imagersize, -2),
    #                         observed_pixel_uncertainty        = 1.0,
    #                         do_optimize_intrinsic_distortions = False,

    #                         do_optimize_intrinsic_core        = False,
    #                         do_optimize_extrinsics            = False,
    #                         do_optimize_frames                = True,

    #                         skip_outlier_rejection            = True,
    #                         skip_regularization               = False,
    #                         verbose                           = True)

    stats = mrcal.optimize( nps.atleast_dims(intrinsics_data, -2),
                            extrinsics_rt_fromref,
                            None, points_subset_positions,
                            None, None,
                            observations_solver, indices_point_camintrinsics_camextrinsics_flags,
                            lensmodel,
                            Npoints_fixed                     = Npoints_fixed_subset,
                            imagersizes                       = nps.atleast_dims(imagersize, -2),
                            observed_pixel_uncertainty        = 1.0,
                            do_optimize_intrinsic_distortions = False,

                            do_optimize_intrinsic_core        = False,
                            do_optimize_extrinsics            = True,
                            do_optimize_frames                = True,

                            skip_outlier_rejection            = True,
                            skip_regularization               = False,
                            verbose                           = False)


    # stats = mrcal.optimize( nps.atleast_dims(intrinsics_data, -2),
    #                         extrinsics_rt_fromref,
    #                         None, points,
    #                         None, None,
    #                         observations, indices_point_camintrinsics_camextrinsics_flags,
    #                         lensmodel,
    #                         Npoints_fixed                     = len(points_fixed),
    #                         imagersizes                       = nps.atleast_dims(imagersize, -2),
    #                         observed_pixel_uncertainty        = 1.0,
    #                         do_optimize_intrinsic_distortions = False,

    #                         do_optimize_intrinsic_core        = True,
    #                         do_optimize_extrinsics            = True,
    #                         do_optimize_frames                = True,

    #                         skip_outlier_rejection            = True,
    #                         skip_regularization               = False,
    #                         verbose                           = True)

    x = stats['x']

    print(f"rms err: {stats['rms_reproj_error__pixels']}")


    rt_rc = mrcal.invert_rt(extrinsics_rt_fromref)
    print(f"camera translation from seed:\n{(rt_rc_orig-rt_rc)[:,3:]}")
    print(f"camera translation from seed mean: {repr(np.mean((rt_rc_orig-rt_rc)[:,3:], axis=-2))}")
    print(f"camera translation from seed mean len: {nps.mag(np.mean((rt_rc_orig-rt_rc)[:,3:], axis=-2))}")

    r_diff = mrcal.compose_rt(mrcal.invert_rt(rt_rc), rt_rc_orig)[:,:3]
    print(f"camera rotation from seed:\n{r_diff}\n{nps.mag(r_diff)*180./np.pi} degrees")

    camera_position_fit = mrcal.invert_rt(extrinsics_rt_fromref)[:,3:]

    range_penalties = x.reshape(len(x)//3,3)[:,2]
    i_points_penalized = range_penalties.nonzero()[0]
    print(f"Solver result: {len(i_points_penalized)} points penalized")





point_labels_map = dict()
for i_point_fixed in range(Npoints_fixed_subset):
    point_labels_map[i_point_fixed + len(points_subset_positions)-Npoints_fixed_subset] = \
        point_labels[i_point_fixed]
mrcal.show_calibration_geometry( extrinsics_rt_fromref,
                                 points=points_subset_positions,
                                 axis_scale=1000,
                                 point_labels = point_labels_map,
                                 hardcopy='/tmp/tst.gp' )

if not args.explore:

    import gnuplotlib as gp

    errx,erry,penalty = nps.transpose(x.reshape(len(x)//3,3))
    # gp.plot( (nps.mag(nps.transpose(nps.cat(errx,erry))), dict(legend='errxy')), (penalty, dict(legend='penalty', y2=1)))
    sys.exit(0)



# # We're exploring!
# import gnuplotlib as gp
# gp.plot(nps.transpose(x.reshape(len(x)//3,3)), legend=np.arange(3))

import IPython
IPython.embed()
sys.exit()



"""
incremental solving
triangulation to find point coords
need visualization to show motion of cameras. To see the zigzag
"""
