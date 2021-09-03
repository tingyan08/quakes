#!/usr/bin/python3

r'''Convert UAVSAR images from raw bits to jpg, and apply the EXIF metadata

This is the pre-processing tool used to convert the data we collect into a form
that's ingestable by Pix4D. The inputs are:

- The images. These may be something standard (like .jpg) or "raw" data. Here
  "raw" means some arbitrary data we ignore, followed by the imager output, 8
  bits per pixel. The output is debayered to produce color data, and some image
  processing is applied to nice-ify the output. This treatment handles arbitrary
  headers in the "raw" data; .pgm works

- INS data from the airplane. A .vnl table of plane poses, synced with the frame
  numbers in the captured images. This is optional. If omitted, no pose data is
  written into the EXIF in the output

- Camera metadata: model numbers and serial numbers of the cameras. Optional. If
  omitted, we don't write it

- Plane-camera transform. These are represented by 3 euler angles. If omitted,
  no pose information is written into the EXIF in the output

- geometric correction. A full 6DOF transform applied to the camera poses, to
  compensate for errors in the other geometric parameters. If omitted, no
  correction is applied.

- camera models: a mrcal model used to provide the intrinsics. If given, these
  are written into the EXIF of the output

'''

import sys
import argparse
import re
import os

def parse_args():

    parser = \
        argparse.ArgumentParser(description = __doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    def comma_separated_list(string, N, t):
        try:
            l = [t(x) for x in string.split(',')]
        except:
            raise argparse.ArgumentTypeError(f"Couldn't parse '{string}' as a comma-separated list of values of type {t}")
        if len(l) != N:
            raise argparse.ArgumentTypeError(f"The argument'{string}' should have {N} values, but instead it had {len(l)} values")
        return l

    parser.add_argument('--image0',
                        type=int,
                        default=0,
                        help='''Start with the Nth matching image in the glob.
                        If omitted, start at the first image''')

    parser.add_argument('--Nimages',
                        type=int,
                        help='''Process N images starting at image0. If omitted,
                        process all images''')

    parser.add_argument('--ins-vnl-plane',
                        type=str,
                        help='''A vnl containing a time-series of poses of the
                        PLANE. We transform these into camera poses, and write
                        those into the EXIF tags. If omitted, we do not write
                        pose information into the EXIF tags''')
    parser.add_argument('--body-manufacturer',
                        type = str,
                        help=r'''Arbitrary string to be written into the
                        'Exif.Image.Make' tag. If omitted, we will not write
                        this tag''')
    parser.add_argument('--body-model',
                        type = str,
                        help=r'''Arbitrary string to be written into the
                        'Exif.Image.Model' tag. If omitted, we will not write
                        this tag''')
    parser.add_argument('--lens-model',
                        type = str,
                        help=r'''Arbitrary string to be written into the
                        'Exif.Photo.LensModel' tag. If omitted, we will not
                        write this tag''')
    parser.add_argument('--body-serial-number',
                        type = str,
                        help=r'''Arbitrary string to be written into the
                        'Exif.Photo.BodySerialNumber' tag. If omitted, we will
                        not write this tag''')
    parser.add_argument('--lens-serial-number',
                        type = str,
                        help=r'''Arbitrary string to be written into the
                        'Exif.Photo.LensSerialNumber' tag. If omitted, we will
                        not write this tag''')
    parser.add_argument('--serial-number',
                        type = str,
                        help=r'''Arbitrary additional string to be written into
                        the "lens model". This primarily exists for pix4d's
                        benefit. Pix4d displays the body and lens models, but no
                        serial numbers. So this is a hack to make the serial
                        number show up in pix4d. If omitted, we will not write
                        this tag''')
    parser.add_argument('--focal-length-nominal-mm',
                        type = float,
                        help=r'''Arbitrary value to be written into the
                        'Exif.Photo.FocalLength' tag. If omitted, we will not
                        write this tag. This is the nominal focal length from
                        the lens manufacturer. The calibrated focal lengths come
                        from the camera model''')
    parser.add_argument('--pixels-per-mm',
                        type = float,
                        help=r'''The pixel pitch of the sensor. If given, this
                        is written into the 'Exif.Photo.FocalPlaneXResolution'
                        and 'Exif.Photo.FocalPlaneYResolution' tags. And it is
                        used to scale the focal length and principal point tags
                        in the EXIF, since those are given in mm. The actual
                        value of this tag doesn't matter: when the downstream
                        tool reads these tags, it will convert the focal lengths
                        and principal point back to units of pixels. But for
                        good hygiene, do provide this value. If omitted, we will
                        not write any tags that need this data, so --model must
                        be omitted also. Similarly, if given, --model must be
                        given also.''')

    parser.add_argument('--camera-roll-deg',
                        type = float,
                        default = 0,
                        help='''The camera roll, relative to the airplane, in
                        degrees. 0deg = "camera is mounted normally"; 90deg =
                        "camera is mounted sideways"''')
    parser.add_argument('--camera-yaw-deg',
                        type = float,
                        default = 0,
                        help='''The camera yaw, relative to the airplane, in
                        degrees. 0deg = "camera is looking in the forward
                        airplane direction"; 90deg = "camera is looking to the
                        left of the airplane"''')
    parser.add_argument('--camera-pitch-deg',
                        type = float,
                        default = 0,
                        help='''The camera pitch, relative to the airplane, in
                        degrees. 0deg = "camera is looking along the
                        horizontal"; 90deg = "camera is looking straight
                        down"''')
    parser.add_argument('--camera-pitch-then-yaw',
                        action='store_true',
                        help='''Specifies whether --camera-pitch-deg is applied
                        before --camera-yaw-deg, or the other way around. By
                        default, we yaw and then pitch (used for the
                        mostly-side-looking SAR fusion geometry). With
                        --camera-pitch-then-yaw we go the other way (used for
                        the mostly down-looking QUAKES geometry)''')
    parser.add_argument('--rt-world-new-world-old',
                        type = lambda s: comma_separated_list(s,6,float),
                        help='''A correction to apply to the world <-> camera
                        transformations. This is a mrcal rt transform (6 values)
                        that maps the original world coord to a corrected one.
                        Must be given as a comma-separated list''')

    parser.add_argument('--rt-cam-new-cam-old',
                        type = lambda s: comma_separated_list(s,6,float),
                        help='''A correction to apply to the world <-> camera
                        transformations. This is a mrcal rt transform (6 values)
                        that maps the original camera coords to corrected ones.
                        Must be given as a comma-separated list''')

    parser.add_argument('--model',
                        type=str,
                        help='''A cameramodel describing the camera used to
                        capture the given images. We write the intrinsics into
                        the EXIF tags. If omitted, we do not write this
                        information into the EXIF tags. Only LENSMODEL_PINHOLE,
                        LENSMODEL_OPENCV4 and LENSMODEL_OPENCV5 are supported''')

    parser.add_argument('--image-dims',
                        type = lambda s: comma_separated_list(s,2,int),
                        help='''A "WIDTH,HEIGHT" pair to specify the image size.
                        This is used only if we're given images that require a
                        raw conversion. This is REQUIRED for raw conversions''')

    parser.add_argument('--outdir',
                        type=str,
                        required=True,
                        help='''The directory where the output images should be
                        written. Must be specified, and must NOT refer to the
                        input images. This tool may overwrite some files, but it
                        will NEVER overwrite its input''')

    parser.add_argument('--saturate-recip',
                        type    = int,
                        default = 65536,
                        help='''We apply a histogram stretch when processing the
                        raw images. To produce visually-pleasing results, we
                        sacrifice a small amount of the histogram at the low and
                        high ends. This produces good contrast in the bulk of
                        the image. The sacrificial amount is specified by this
                        argument. For instance "--saturate-recip 100" means the
                        darkest 1/100th of the pixels end up at 0, and the
                        brightest 1/100th end up at 255. In practice, a much
                        bigger value than 100 is desired. We default to 65536''')

    parser.add_argument('--camera-poses-only',
                        action = 'store_true',
                        help='''If given, we ONLY produce the vnl with the
                        camera poses. We do NOT do any raw conversion. We do NOT
                        write any images. We do NOT write EXIF tags to
                        anything''')

    parser.add_argument('--jobs', '-j',
                        type=int,
                        default=1,
                        help='''parallelize the processing JOBS-ways. This is
                        like Make, except you're required to explicitly specify
                        a job count. If omitted, no parallelization happens''')

    parser.add_argument('images',
                        type=str,
                        nargs='+',
                        help='''Globs containing images we're converting. jpg
                        and png and tiff images are assumed to already be
                        usable. Everything else is run through a raw
                        converter''')

    args = parser.parse_args()
    return args

args = parse_args()




import numpy as np
import numpysane as nps
import glob
import pyproj
import pyexiv2
import fractions
import mrcal
import vnlog
import cv2
import shutil
import multiprocessing
import signal
import _raw_convert


if args.model is not None:
    try:
        m = mrcal.cameramodel(args.model)
    except:
        print(f"Couldn't parse cameramodel from '{args.model}'", file=sys.stderr)
        sys.exit(1)

    lensmodel = m.intrinsics()[0]
    if not (lensmodel == 'LENSMODEL_PINHOLE' or
            lensmodel == 'LENSMODEL_OPENCV4' or
            lensmodel == 'LENSMODEL_OPENCV5'):
        print(f"I only support LENSMODEL_PINHOLE, LENSMODEL_OPENCV4, LENSMODEL_OPENCV5, but read '{lensmodel}' from '{args.model}'", file=sys.stderr)
        sys.exit(1)

    if args.pixels_per_mm is None:
        print(f"Both --model and --pixels-per-mm should be given, or neither should be", file=sys.stderr)
        sys.exit(1)

else:
    if args.pixels_per_mm is not None:
        print(f"Both --model and --pixels-per-mm should be given, or neither should be", file=sys.stderr)
        sys.exit(1)

##################### copy of functions from sfmviz.py. Please consolidate
# https://github.jpl.nasa.gov/kogan/sfmviz

# Defines the local cartesian coordinate system. Before use, MUST be filled-in
# by proj_context__init()
proj_context = None

def proj_context__init(latlon,
                       allow_reinit = False,
                       recenter     = True):
    r'''Initializes our locally-planar coordinate system at the given latlon

Used by ne_from_latlon() and latlon_from_ne(). This function MUST be called
before either of those

    '''

    global proj_context

    if proj_context is not None:
        if not allow_reinit:
            raise Exception("Trying to initialize an already-initialized proj_context")

    zone = 1 + int((latlon[1] + 180) // 6)
    proj_context = dict(proj = \
                        pyproj.Proj(proj           = 'utm',
                                    zone           = zone,
                                    ellps          = 'WGS84',
                                    preserve_units = False # output in meters
                                    ))

    e,n = proj_context['proj'](latlon[1], latlon[0])
    if recenter:
        proj_context['ne0'] = np.array((n,e))
    else:
        proj_context['ne0'] = (0,0)


def ne_from_latlon(latlon, get_gradients=False):
    '''Compute translated UTM (northing,easting) from a lat,lon pair

Reports (northing,easting) in meters in a common UTM coordinate system. The
local coordinate system is defined by calling proj_context__init(), which MUST
be done before calling this function

    '''

    global proj_context
    if proj_context is None:
        raise Exception("Need to initialize the context by calling proj_context__init() before computing coord transformations")

    lat = latlon[..., 0]
    lon = latlon[..., 1]
    en  = proj_context['proj'](lon,lat)
    e,n = en

    ne = nps.glue( nps.dummy(np.array(n),-1),
                   nps.dummy(np.array(e),-1),
                   axis=-1 ) - proj_context['ne0']

    if not get_gradients:
        return ne

    # gradients
    #
    # I'm using pyproj, which has some gradient support
    # (proj_context['proj'].get_factors() returns something relevant-looking).
    # But there's no clear way to convert this to a full gradient matrix, so I
    # don't even bother. I use forward differences, and call it good.
    dlatlon = 1e-6
    dne_dlatlon = np.zeros(latlon.shape + (2,), dtype=float)
    den_dlat = np.array(proj_context['proj'](lon,        lat+dlatlon)) - en
    den_dlon = np.array(proj_context['proj'](lon+dlatlon,lat        )) - en
    dne_dlatlon[...,0,0] = den_dlat[...,1] / dlatlon
    dne_dlatlon[...,1,0] = den_dlat[...,0] / dlatlon
    dne_dlatlon[...,0,1] = den_dlon[...,1] / dlatlon
    dne_dlatlon[...,1,1] = den_dlon[...,0] / dlatlon

    return ne, dne_dlatlon


def latlon_from_ne(ne):
    '''Compute lat,lon from a translated UTM (northing,easting)

Ingests a (northing,easting) in meters in a common UTM coordinate system. The
local coordinate system is defined by calling proj_context__init(), which MUST
be done before calling this function

    '''

    global proj_context
    if proj_context is None:
        raise Exception("Need to initialize the context by calling proj_context__init() before computing coord transformations")

    lon,lat = proj_context['proj'](ne[..., 1] + proj_context['ne0'][1],
                                   ne[..., 0] + proj_context['ne0'][0], inverse=True)

    return nps.glue( nps.dummy(np.array(lat),-1),
                     nps.dummy(np.array(lon),-1),
                     axis=-1 )


def R_world_cam__from_pix4d_rpy_plane(rpy_plane,
                                      camera_roll_deg       = 0,
                                      camera_yaw_deg        = 0,
                                      camera_pitch_deg      = 0,
                                      camera_pitch_then_yaw = False):
    r'''Computes a rotation from pix4d-style euler angles

There are many interpretations of "roll,pitch,yaw". This function follows the
Pix4D conventions, defined on their support page:

  https://support.pix4d.com/hc/en-us/articles/202558969

Specifically, in the document linked on that page, and with a copy on our
server:

  https://aargh.jpl.nasa.gov/proj/uavsar/docs/Pix4D_Yaw_Pitch_Roll_Omega_to_Phi_Kappa_angles_and_conversion.pdf

The "world" frame is "north,east,down"
The "body"  frame (from that pdf) is "forward,right,down"

I make an additional transformation to use mrcal's camera coordinate
system, which is (right,down,forward).

We return a rotation to map mrcal-style camera coords (right,down,forward) to
world coords (north,east,down)

    '''

    r,p,y = nps.transpose(rpy_plane)
    sy,cy = np.sin(y),np.cos(y)
    sp,cp = np.sin(p),np.cos(p)
    sr,cr = np.sin(r),np.cos(r)

    # I want to do this:
    #   Ry = np.array((( cy, -sy, 0.),
    #                  ( sy,  cy, 0.),
    #                  ( 0.,  0., 1.)))
    #
    #   Rp = np.array((( cp, 0., sp),
    #                  ( 0., 1., 0.),
    #                  (-sp, 0., cp)))
    #
    #   Rr = np.array(((1., 0.,  0.),
    #                  (0., cr, -sr),
    #                  (0., sr,  cr)))
    #
    # But each R should have shape (...,3,3), so I need to do more typing:
    Ry = np.zeros( y.shape + (3,3), dtype=float)
    Ry[...,2,2] = 1.
    Ry[...,0,0] = cy
    Ry[...,0,1] = -sy
    Ry[...,1,0] = sy
    Ry[...,1,1] = cy

    Rp = np.zeros( p.shape + (3,3), dtype=float)
    Rp[...,1,1] = 1.
    Rp[...,0,0] = cp
    Rp[...,0,2] = sp
    Rp[...,2,0] = -sp
    Rp[...,2,2] = cp

    Rr = np.zeros( r.shape + (3,3), dtype=float)
    Rr[...,0,0] = 1.
    Rr[...,1,1] = cr
    Rr[...,1,2] = -sr
    Rr[...,2,1] = sr
    Rr[...,2,2] = cr

    R_world_body = nps.matmult(Ry,Rp,Rr)

    # The rpy we just processed is from a PLANE. We need an extra transformation
    # to remap this into CAMERA coords. We have another set of (yaw,pitch,roll),
    # describing the pose of the camera in respect to the plane. In my one test
    # case so far I have the cameras looking ~90deg to the left and are mounted
    # sideways (roll ~ 90deg), and are looking some amount down (non-zero
    # pitch). I need to incorporate the rotations AND take into account the
    # different coord systems. The plane (body) coordinate system is
    # (forward,right,down), while the mrcal camera coord system is
    # (right,down,forward). Testing in my one case so far tells me that for sp =
    # sin(pitch), cp = sin(pitch) I should get
    #
    #   R_cam_body = np.array((
    #       # cam-right in plane coords
    #       (0,  sp, cp),
    #       # cam-down in plane coords
    #       (-1., 0., 0.),
    #       # cam-forward in plane coords
    #       (0., -cp, sp)))
    #
    # I engineer a sequence of Euler angle transformations that produces this
    # result, and I validate it using this script:
    #
    #    import numpy as np
    #    import numpysane as nps
    #    import sympy
    #
    #    sp,cp = sympy.symbols(('sp','cp'))
    #    sy,cy = sympy.symbols(('sy','cy'))
    #    sr,cr = sympy.symbols(('sr','cr'))
    #
    #    # yaw and roll = 90deg (camera is sideways, looking out to the side)
    #    sy = 1
    #    cy = 0
    #
    #    sr = 1
    #    cr = 0
    #
    #    R_cam_body_reorder = np.array(((0,1,0),
    #                                   (0,0,1),
    #                                   (1,0,0)))
    #
    #    Ryaw   = np.array((( cy, 0, sy),
    #                       (  0, 1,  0),
    #                       (-sy, 0, cy)))
    #    Rpitch = np.array(((1,  0,  0),
    #                       (0, cp,-sp),
    #                       (0, sp, cp)))
    #    Rroll  = np.array((( cr, sr, 0),
    #                       (-sr, cr, 0),
    #                       (  0,  0, 1)))
    #
    #    R_cam_body = nps.matmult(Rroll, Rpitch, Ryaw, R_cam_body_reorder)
    #
    #    print(repr(R_cam_body))
    #
    # It reports
    #
    #   array([[0, sp, cp],
    #          [-1, 0, 0],
    #          [0, -cp, sp]], dtype=object)
    #
    # Which matches the desired R_cam_body above. Without hardcoding the
    # yaw,roll: and I to get:
    sr = np.sin(camera_roll_deg  * np.pi/180.0)
    cr = np.cos(camera_roll_deg  * np.pi/180.0)
    sp = np.sin(camera_pitch_deg * np.pi/180.0)
    cp = np.cos(camera_pitch_deg * np.pi/180.0)
    sy = np.sin(camera_yaw_deg   * np.pi/180.0)
    cy = np.cos(camera_yaw_deg   * np.pi/180.0)

    # camera (right,down,forward) <- plane (forward,right,down)
    R_cam_body_reorder = np.array(((0,1,0),
                                   (0,0,1),
                                   (1,0,0)))

    # Initially, the camera is oriented horizontally, looking ahead, along the
    # main axis of the plane.
    #
    # We yaw around the vertical axis (camera y); assuming no
    # --camera-pitch-then-yaw
    Ryaw   = np.array((( cy, 0, sy),
                       (  0, 1,  0),
                       (-sy, 0, cy)))

    # Then we pitch around the camera's new horizontal axis (camera x); assuming
    # no --camera-pitch-then-yaw
    Rpitch = np.array(((1,  0,  0),
                       (0, cp,-sp),
                       (0, sp, cp)))

    # Finally we rotate the camera along its view axis (camera z)
    Rroll  = np.array((( cr, sr, 0),
                       (-sr, cr, 0),
                       (  0,  0, 1)))

    if camera_pitch_then_yaw:
        R_cam_body = nps.matmult(Rroll, Ryaw, Rpitch, R_cam_body_reorder)
    else:
        R_cam_body = nps.matmult(Rroll, Rpitch, Ryaw, R_cam_body_reorder)

    return nps.matmult(R_world_body, nps.transpose(R_cam_body))


def Rt_world_cam__compute(rpydeg_latlonalt,
                          camera_roll_deg       = 0,
                          camera_yaw_deg        = 0,
                          camera_pitch_deg      = 0,
                          camera_pitch_then_yaw = False):

    r'''Computes a transformation from EXIF metadata

Returns a transformation (in an (Ncameras,4,3) array) to map mrcal-style camera
coords (right,down,forward) to world coords (north,east,down).

    '''

    rpy = rpydeg_latlonalt[:,:3] * np.pi/180.

    # shape (...,3,3)
    R_world_cam = R_world_cam__from_pix4d_rpy_plane(rpy,
                                                    camera_roll_deg       = camera_roll_deg,
                                                    camera_yaw_deg        = camera_yaw_deg,
                                                    camera_pitch_deg      = camera_pitch_deg,
                                                    camera_pitch_then_yaw = camera_pitch_then_yaw)

    lat,lon,alt = nps.transpose(rpydeg_latlonalt[:, 3:6])

    # shape (...,1,3)
    ned = nps.glue( # shape (...,1,2)
                    ne_from_latlon(# shape (...,1,2)
                                   nps.glue(# shape (...,1,1)
                                       nps.mv( lat,-1,-3),
                                       nps.mv( lon,-1,-3),
                                       axis=-1)),
                    # shape (...,1,1)
                    nps.mv(-alt,-1,-3),
                    axis = -1 )

    Rt_world_cam = nps.glue(R_world_cam, ned,
                            axis = -2)
    return Rt_world_cam


def parse_ins_vnl_plane(ins_vnl_plane, image0 = 0, Nimages = None, filenames = None):

    x,list_keys,dict_key_index = vnlog.slurp(ins_vnl_plane)

    colidx = (dict_key_index['roll_deg'],
              dict_key_index['pitch_deg'],
              dict_key_index['heading_deg'],
              dict_key_index['latitude'],
              dict_key_index['longitude'],
              dict_key_index['altitude_ft'] )

    if filenames is None:

        x = x[image0:]
        if Nimages is not None:
            x = x[:Nimages]

        rpydeg_latlonalt = x[:, colidx ]

    else:
        # The set of frames described the the ins_vnl_plane file may be
        # different from the set of frames in the globs the user requested. I
        # make sure that we have an ins entry for each requested frame. And I
        # construct the rpydeg_latlonalt array such that the nth row of this
        # array represents the nth frame
        rpydeg_latlonalt = np.zeros((len(filenames),6), dtype=float)

        frame_numbers_from_ins_vnl_plane = x[:,dict_key_index['frame_index']].astype(int)

        row_from_frameindex = dict()
        for row in range(len(frame_numbers_from_ins_vnl_plane)):
            row_from_frameindex[frame_numbers_from_ins_vnl_plane[row]] = row


        frame_numbers_from_globs = np.array([int(os.path.basename(f).split('_')[0]) for f in filenames])
        for i in range(len(filenames)):
            frameindex = int(os.path.basename(filenames[i]).split('_')[0])

            try:
                row = row_from_frameindex[frameindex]
            except:
                raise Exception(f"The user requested frame {frameinde} (from file {filenames[i]}), but this wasn't available in {ins_vnl_plane}")

            rpydeg_latlonalt[i,:] = x[row, colidx]

    # altitude: ft -> m
    rpydeg_latlonalt[:,5] *= 12.*2.54/100.

    return rpydeg_latlonalt

#################################################





def decompose_pose(Rt_world_cam):

    # "world" is (N,E,down)

    latlonalt_opkdeg = np.zeros( Rt_world_cam.shape[:-2] + (6,), dtype=float)

    ned = Rt_world_cam[...,3,:]

    latlonalt_opkdeg[..., :2] = latlon_from_ne(ned[..., :2])

    # altitude is measured UP, so I invert the DOWN value
    latlonalt_opkdeg[..., 2] = -ned[..., 2]

    # Docs are here:
    #
    #   https://support.pix4d.com/hc/en-us/articles/202558969
    #
    # Specifically, in the document linked on that page, and with a copy on our
    # server:
    #
    #   https://aargh.jpl.nasa.gov/proj/uavsar/docs/Pix4D_Yaw_Pitch_Roll_Omega_to_Phi_Kappa_angles_and_conversion.pdf
    R_world_cam = Rt_world_cam[..., :3,:]

    # The pdf document above describes a transformation C_EB.
    #
    # I'm assuming the Earth is locally flat, so the coord system E is always
    # aligned with the ground. Docs say this coord system is (E,N,up)
    R_E_world = np.array(((0,1, 0),
                          (1,0, 0),
                          (0,0,-1)))

    # The coord system B is aligned with the camera. It is (right,up,back)
    R_cam_B = np.array(((1, 0, 0),
                        (0,-1, 0),
                        (0, 0,-1)))

    R_EB = nps.matmult(R_E_world, R_world_cam, R_cam_B)

    # The pdf document defines C_EB (same as my R_EB) as a function of
    # (omega,phi,kappa). I decompose

    # I assume that phi is in [-180,180], so cos(phi)>=0
    sp = R_EB[..., 0,2]
    p = np.arcsin(sp)
    k = np.arctan2(-R_EB[..., 0,1], R_EB[..., 0,0])
    o = np.arctan2(-R_EB[..., 1,2], R_EB[..., 2,2])

    latlonalt_opkdeg[..., 3] = o*180./np.pi
    latlonalt_opkdeg[..., 4] = p*180./np.pi
    latlonalt_opkdeg[..., 5] = k*180./np.pi
    return latlonalt_opkdeg


def process_image(filename,

                  # Anything that is None is not set
                  latlonalt_opkdeg        = None,
                  model_intrinsics        = None,
                  body_manufacturer       = None,
                  body_model              = None,
                  lens_model              = None,
                  body_serial_number      = None,
                  lens_serial_number      = None,
                  serial_number           = None,
                  focal_length_nominal_mm = None,
                  pixels_per_mm           = None):

    def rational(x, denominator=65536):
        # It is completely ridiculous that I have to do this
        numerator = int(x*denominator)
        return fractions.Fraction(numerator,denominator)
    def GPSCoordinate(x):
        # It is completely ridiculous that I have to do all this
        ax = abs(x)
        d  = int(ax)
        m  = int(60*(ax - d))
        s  = 3600*(ax - d) - 60*m
        return \
            [ fractions.Fraction(d,1),
              fractions.Fraction(m,1),
              rational(s) ]


    tags = dict()

    if model_intrinsics is not None:
        lensmodel, intrinsics = model_intrinsics.intrinsics()
        if not (lensmodel == 'LENSMODEL_PINHOLE' or
                lensmodel == 'LENSMODEL_OPENCV4' or
                lensmodel == 'LENSMODEL_OPENCV5'):
            raise Exception("I only support LENSMODEL_{PINHOLE,OPENCV4,OPENCV5}")

        fxy = intrinsics[0:2]
        cxy = intrinsics[2:4]
        if lensmodel == 'LENSMODEL_PINHOLE':
            r1,r2,t1,t2,r3 = 0,0,0,0,0
        elif lensmodel == 'LENSMODEL_OPENCV4':
            r1,r2,t1,t2 = intrinsics[4:]
            r3 = 0
        elif lensmodel == 'LENSMODEL_OPENCV5':
            r1,r2,t1,t2,r3 = intrinsics[4:]

        tags['Xmp.Camera.PerspectiveDistortion'] = ','.join(str(x) for x in (r1,r2,r3,t1,t2))

        if pixels_per_mm is not None:

            fxy_mean = (fxy[0] + fxy[1])/2.

            if np.abs(fxy[0] - fxy[1]) > 1e-6:
                print(f"## WARNING: Pix4d assumes fx==fy, but this model has unequal fxy: {fxy}. I write the mean into the EXIF: {fxy_mean}")

            tags['Xmp.Camera.PrincipalPoint']         = f'{cxy[0]/pixels_per_mm},{cxy[1]/pixels_per_mm}'
            tags['Xmp.Camera.PerspectiveFocalLength'] = str(fxy_mean / pixels_per_mm)
            tags['Exif.Photo.FocalPlaneXResolution']  = rational(pixels_per_mm)
            tags['Exif.Photo.FocalPlaneYResolution']  = rational(pixels_per_mm)

    if body_manufacturer       is not None: tags['Exif.Image.Make']             = body_manufacturer
    if body_model              is not None: tags['Exif.Image.Model']            = body_model
    if lens_model              is not None: tags['Exif.Photo.LensModel']        = lens_model
    if body_serial_number      is not None: tags['Exif.Photo.BodySerialNumber'] = body_serial_number
    if lens_serial_number      is not None: tags['Exif.Photo.LensSerialNumber'] = lens_serial_number
    if focal_length_nominal_mm is not None: tags['Exif.Photo.FocalLength']      = rational(focal_length_nominal_mm)

    if serial_number is not None:
        if 'Exif.Photo.LensModel' not in tags:
            tags['Exif.Photo.LensModel'] = serial_number
        else:
            tags['Exif.Photo.LensModel'] += ': ' + serial_number

    if latlonalt_opkdeg is not None:
        lat,lon,alt = latlonalt_opkdeg[:3]

        tags['Exif.GPSInfo.GPSLatitude']     = GPSCoordinate(abs(lat))
        tags['Exif.GPSInfo.GPSLongitude']    = GPSCoordinate(abs(lon))
        tags['Exif.GPSInfo.GPSAltitude']     = rational(alt)
        tags['Exif.GPSInfo.GPSLatitudeRef']  = 'N' if lat >= 0 else 'S'
        tags['Exif.GPSInfo.GPSLongitudeRef'] = 'E' if lon >= 0 else 'W'
        tags['Exif.GPSInfo.GPSAltitudeRef']  = '0' # above sea level

        # Indicate very poor GPS accuracy
        tags['Xmp.Camera.GPSXYAccuracy']     = "1000"
        tags['Xmp.Camera.GPSZAccuracy']      = "1000"


    # Split the filename into = d + f + e (directory, basename, extension)
    d,f = os.path.split(filename)
    f,e = os.path.splitext(f)

    if is_raw_image(filename):
        # I convert the raw, write the .jpg to disk, and then update the exif of
        # the file on disk
        if 0:
            image = raw_convert__prototype(filename,
                                           *args.image_dims,
                                           unbayer = True)

        else:
            image = _raw_convert.raw_convert(filename,
                                             *args.image_dims,
                                             saturate_recip = args.saturate_recip)

        filename = f"{args.outdir}/{f}.jpg"

        cv2.imwrite(filename,
                    image,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])

    else:
        # I already have a processed image on disk. I copy it to its final
        # location, and update the exif there
        filename_out = f"{args.outdir}/{f}{e}"
        shutil.copy(filename, filename_out)
        filename = filename_out

    if not tags:
        # no EXIF tags to write. I'm done
        return filename

    exif = pyexiv2.ImageMetadata(filename)
    exif.read()

    # Required to be able to touch Xmp.Camera tags
    try:
        pyexiv2.register_namespace('https://support.pix4d.com/hc/en-us/articles/205732309-EXIF-and-XMP-tag-information-read-by-Pix4D-Desktop/', 'Camera')
    except:
        pass

    tags.update({ 'Xmp.Camera.ModelType':                'perspective',
                  'Exif.Photo.FocalPlaneResolutionUnit': 4, # "mm"
                 })

    for key in tags:
        if key in exif:
            print(f"## WARNING: key '{key}' already exists in the metadata for '{filename}'. Modifying existing tags is an untested case, so I do nothing for now")
            continue

        if   re.match('Xmp',  key): tag = pyexiv2.XmpTag (key, tags[key])
        elif re.match('Exif', key): tag = pyexiv2.ExifTag(key, tags[key])
        else: raise Exception("I only know how to make EXIF and XMP tags")
        exif[key] = tag

    exif.write()

    return filename


def raw_convert__prototype(filename, width, height,
                           unbayer = True):

    r'''Read raw bits, convert to jpg

    With some simple image processing along the way. This is the reference
    implementation for _raw_convert.raw_convert(). The two functions do the same
    thing, but _raw_convert.raw_convert() is written in C, with several
    optimizations in mind, so it is much faster. THIS function is good for
    testing and prototyping

    '''

    datasize_expected = width*height

    filesize = os.path.getsize(filename)

    if filesize <= datasize_expected:
        raise Exception(f"I expected {datasize_expected} bytes of pixel data, but the given file only has {filesize} bytes in it. That is too small")

    image = np.memmap(filename,
                      dtype  = np.uint8,
                      mode   = 'r',
                      offset = filesize - datasize_expected,
                      shape  = (height,width),
                      order  = 'C')

    if unbayer:
        image = cv2.cvtColor(image, cv2.COLOR_BayerRG2RGB)

        # Histogram equalization. Too aggressive, so I leave it off by default
        if 0:
            for i in range(3):
                image[...,i] = cv2.equalizeHist(image[...,i])
        else:
            # Grayworld, some sort of stretch equalization

            for i in range(3):

                # First, grayworld. This just means "scale the intensities to
                # move the mean to the center"
                #
                # cv2.normalize() gives me the saturated scaling with 8-bit
                # unsigned values
                x  = image[..., i]
                xr = x.ravel()

                cv2.normalize(src       = xr,
                              dst       = xr,
                              # *len() to make norm_l1 = mean
                              alpha     = 128*len(xr),
                              norm_type = cv2.NORM_L1,
                              dtype     = cv2.CV_8U)

                # The input was written back to the array. The mean was scaled
                # to 128. Grayworld is done

                # I linearly stretch the intensities such that the histogram
                # covers my full range, sacrificing some stragglers at the low
                # and high ends
                xmin = np.percentile(xr,       100./args.saturate_recip)
                xmax = np.percentile(xr, 100 - 100./args.saturate_recip)

                np.clip(xr, xmin, xmax, out=xr)
                cv2.normalize(src = xr,
                              dst = xr,
                              alpha     = 0,
                              beta      = 255,
                              norm_type = cv2.NORM_MINMAX,
                              dtype     = cv2.CV_8U)

                # I shoudn't need to do this, but this doesn't work otherwise.
                # Earlier I did xr = x.ravel(). I expected xr to be a view into
                # x, but for some reason it's making a copy, so I need to copy
                # the processed results back. I have no idea why it's making a
                # copy.
                x[...] = xr.reshape(x.shape)

    return image


def is_raw_image(f):
    try:
        pattern = is_raw_image.pattern
    except:
        pattern = is_raw_image.pattern = \
            re.compile('.*\.(jpg|jpeg|png|tif|tiff)$', re.I)
    return not re.match(pattern, f)





if args.model is not None:
    model_intrinsics = mrcal.cameramodel(os.path.expanduser(args.model))
else:
    model_intrinsics = None

expanded_globs = [glob.glob(os.path.expanduser(g)) for g in args.images]
for i in range(len(expanded_globs)):
    if len(expanded_globs[i]) == 0:
        raise Exception(f"Glob {args.images[i]} matched nothing")
filenames = sorted( [f for g in expanded_globs for f in g] )
filenames = filenames[args.image0:]
if args.Nimages is not None:
    filenames = filenames[:args.Nimages]

if args.image_dims is None and any([is_raw_image(f) for f in filenames]):
    r0 = [f for f in filenames if is_raw_image(f)][0]
    raise Exception(f"Some images are raw, but --image-dims not given. First raw image: {r0}")

for f in filenames:
    d0 = os.path.realpath( os.path.abspath( args.outdir ))
    d1 = os.path.realpath( os.path.abspath( os.path.split(f)[0] ))
    if d0 == d1:
        raise Exception("--outdir may not be a path that contains any of the input files. I want to make sure to never overwrite the input")

os.makedirs(args.outdir, mode=0o775, exist_ok = True)



if args.ins_vnl_plane is not None:

    print("# filename latitude longitude altitude_m omega_deg phi_deg kappa_deg")

    rpydeg_latlonalt__plane = \
        parse_ins_vnl_plane(args.ins_vnl_plane,
                            args.image0,
                            args.Nimages,
                            filenames)

    proj_context__init( # lat,lon
                        (rpydeg_latlonalt__plane[0,3], rpydeg_latlonalt__plane[0,4] ))

    # shape (Ncam, 4,3)
    Rt_world_cam = \
        Rt_world_cam__compute(rpydeg_latlonalt__plane,
                              camera_roll_deg       = args.camera_roll_deg,
                              camera_yaw_deg        = args.camera_yaw_deg,
                              camera_pitch_deg      = args.camera_pitch_deg,
                              camera_pitch_then_yaw = args.camera_pitch_then_yaw)

    # Apply pose corrections
    if args.rt_world_new_world_old is not None:
        Rt_world_cam = mrcal.compose_Rt( mrcal.Rt_from_rt(np.array(args.rt_world_new_world_old)),
                                         Rt_world_cam )
    if args.rt_cam_new_cam_old is not None:
        Rt_world_cam = mrcal.compose_Rt( Rt_world_cam,
                                         mrcal.invert_Rt(mrcal.Rt_from_rt(np.array(args.rt_cam_new_cam_old))) )

    latlonalt_opkdeg = decompose_pose(Rt_world_cam)

else:
    latlonalt_opkdeg = None





def output_line(f,i):
    if latlonalt_opkdeg is not None:
        print(' '.join( [f] + [str(x) for x in latlonalt_opkdeg[i]]))
    else:
        print(f"## Wrote {f}")


if args.camera_poses_only:
    for i in range(len(filenames)):
        if is_raw_image(filenames[i]):
            # Split the filename into = d + f + e (directory, basename, extension)
            d,f = os.path.split(filenames[i])
            f,e = os.path.splitext(f)
            output_line(f"{args.outdir}/{f}.jpg", i)
        else:
            output_line(filenames[i], i)

else:

    def process_one(i):
        filename_out = \
            process_image(filenames[i],
                          latlonalt_opkdeg[i] if latlonalt_opkdeg is not None else None,
                          model_intrinsics,
                          args.body_manufacturer,
                          args.body_model,
                          args.lens_model,
                          args.body_serial_number,
                          args.lens_serial_number,
                          args.serial_number,
                          args.focal_length_nominal_mm,
                          args.pixels_per_mm)

        output_line(filename_out,i)



    if args.jobs <= 1:
        for i in range(len(filenames)):
            process_one(i)

    else:
        # Parallelize!

        # weird business to handle weird signal handling in multiprocessing. I want
        # things like the user hitting C-c to work properly. So I ignore SIGINT for the
        # children. And I want the parent's blocking wait for results to respond to
        # signals. Which means map_async() instead of map(), and wait(big number)
        # instead of wait()
        signal_handler_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal_handler_sigint)

        pool = multiprocessing.Pool(args.jobs)
        try:
            mapresult = pool.map_async(process_one, range(len(filenames)))

            # like wait(), but will barf if something goes wrong. I don't actually care
            # about the results
            mapresult.get(100000000)
        except:
            pool.terminate()

        pool.close()
        pool.join()
