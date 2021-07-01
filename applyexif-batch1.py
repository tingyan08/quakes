#!/usr/bin/python3

import sys
import os
import re
import numpy as np
import numpysane as nps
import glob
import pyproj
import pyexiv2
import fractions
import mrcal




##################### functions from sfmviz.py
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


def R_world_cam__from_pix4d_rpy(rpy, have_custom_pose):
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

    r,p,y = nps.transpose(rpy)
    sr,cr = np.sin(r),np.cos(r)
    sp,cp = np.sin(p),np.cos(p)
    sy,cy = np.sin(y),np.cos(y)

    # I want to do this:
    #   Rr = np.array(((1., 0.,  0.),
    #                  (0., cr, -sr),
    #                  (0., sr,  cr)))
    #
    #   Rp = np.array((( cp, 0., sp),
    #                  ( 0., 1., 0.),
    #                  (-sp, 0., cp)))
    #
    #   Ry = np.array((( cy, -sy, 0.),
    #                  ( sy,  cy, 0.),
    #                  ( 0.,  0., 1.)))
    #
    # But each R should have shape (...,3,3), so I need to do more typing:
    Rr = np.zeros( r.shape + (3,3), dtype=float)
    Rr[...,0,0] = 1.
    Rr[...,1,1] = cr
    Rr[...,1,2] = -sr
    Rr[...,2,1] = sr
    Rr[...,2,2] = cr

    Rp = np.zeros( p.shape + (3,3), dtype=float)
    Rp[...,1,1] = 1.
    Rp[...,0,0] = cp
    Rp[...,0,2] = sp
    Rp[...,2,0] = -sp
    Rp[...,2,2] = cp

    Ry = np.zeros( y.shape + (3,3), dtype=float)
    Ry[...,2,2] = 1.
    Ry[...,0,0] = cy
    Ry[...,0,1] = -sy
    Ry[...,1,0] = sy
    Ry[...,1,1] = cy

    R_world_body = nps.matmult(Ry,Rp,Rr)

    if not have_custom_pose:

        # Usual logic. rpy is from a drone, and describes the pose of the
        # CAMERA. The camera coords they're using are (forward,right,down),
        # while I want (right,down,forward), so we transform

        R_body_cam = np.array(((0,0,1),
                               (1,0,0),
                               (0,1,0)))

        return nps.matmult(R_world_body, R_body_cam)


    # Weird logic. rpy is from hand-wavy ins data from an airplane, and
    # describes the pose of the PLANE. The cameras are looking ~90deg to the
    # left (hardcoded below) and ~45deg down (adjustable via the "tilt" var).
    # AND the cameras are mounted sideways. I guesstimate the pose and the
    # method for interpreting the euler angles, until it sorta looks right

    # I have the world-plane transform. Need the plane-cam transform. Plane
    # (body) coordinate system is (forward,right,down)
    tilt = 35. * np.pi/180.0 # how much below the horizontal
    s = np.sin(tilt)
    c = np.cos(tilt)

    R_cam_body = np.array((
        # cam-right in plane coords
        (0,  s, c),

        # cam-down in plane coords
        (-1., 0., 0.),

        # cam-forward in plane coords
        (0., -c, s)))

    return nps.matmult(R_world_body, nps.transpose(R_cam_body))


def Rt_world_cam__compute(rpydeg_latlonalt_date, have_custom_pose):
    r'''Computes a transformation from EXIF metadata

Returns a transformation (in an (Ncameras,4,3) array) to map mrcal-style camera
coords (right,down,forward) to world coords (north,east,down).

    '''

    rpy = rpydeg_latlonalt_date[:,:3] * np.pi/180.

    # shape (...,3,3)
    R_world_cam = R_world_cam__from_pix4d_rpy(rpy, have_custom_pose)

    lat,lon,alt = nps.transpose(rpydeg_latlonalt_date[:, 3:6])

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


def parse_posefile(posefile, imageglob, first_image = 0, Nimages = None):
    with open(os.path.expanduser(posefile), 'r') as f:
        f.readline()
        f.readline()
        x = np.loadtxt(f)

        x = x[first_image:]

        if Nimages is not None:
            x = x[:Nimages]

        frame         = x[:,0].astype(int)
        lat           = x[:,2]
        lon           = x[:,3]
        alt           = x[:,4] * 12.*2.54/100.
        heading_plane = x[:,5]
        pitch_plane   = x[:,7]
        roll_plane    = x[:,8]

        # All the logic for parsing pitch/roll/heading is in
        # R_world_cam__from_pix4d_rpy()
        rpydeg_latlonalt_date = \
            nps.transpose(nps.cat( roll_plane, pitch_plane, heading_plane,
                                   lat, lon, alt,
                                   alt*0 ))

        filenames = sorted(glob.glob(os.path.expanduser(imageglob)))

        filenames = filenames[first_image:]
        if Nimages is not None:
            filenames = filenames[:Nimages]

        frame_numbers = [int(os.path.basename(f).split('_')[0]) for f in filenames]

        if np.max(np.abs(frame - np.array(frame_numbers))) > 0:
            raise Exception(f"The images matched by the glob '{imageglob}' don't match the frame numbers in ins file '{posefile}'")
    return filenames, rpydeg_latlonalt_date

#################################################





def decompose_pose(Rt_world_cam):
    ned = Rt_world_cam[...,3,:]

    lat,lon = nps.mv(latlon_from_ne(ned[..., :2]),
                     -1,0)

    # altitude is measured UP, so I invert the DOWN value
    alt     = -ned[..., 2]


    # I now reverse the R_world_cam__from_pix4d_rpy() function
    R_world_cam = Rt_world_cam[..., :3,:]

    R_body_cam = np.array(((0,0,1),
                           (1,0,0),
                           (0,1,0)))

    R_world_body = nps.matmult( R_world_cam, nps.transpose(R_body_cam))

    r'''I symbolically evaluate the key transform in R_world_cam__from_pix4d_rpy using sympy:

        #!/usr/bin/python3

        import sys
        import numpy as np
        import numpysane as nps
        import sympy

        sr,cr = sympy.symbols(('sr','cr'))
        sp,cp = sympy.symbols(('sp','cp'))
        sy,cy = sympy.symbols(('sy','cy'))


        Rr = np.array(((1, 0 ,  0),
                       (0, cr , -sr),
                       (0, sr,  cr)))

        Rp = np.array(((cp , 0, sp),
                       (0 ,  1, 0),
                       (-sp, 0, cp)))

        Ry = np.array(((cy, -sy, 0),
                       (sy, cy,  0),
                       (0,  0,   1)))

        print(repr(nps.matmult(Ry,Rp,Rr)))


    ---> array([[cp*cy, -cr*sy + cy*sp*sr, cr*cy*sp + sr*sy],
                [cp*sy,  cr*cy + sp*sr*sy, cr*sp*sy - cy*sr],
                [-sp,    cp*sr,            cp*cr]],           dtype=object)


    and I can then use this to back-out the Euler angles

    '''

    # I assume that pitch ~ 0, so sp ~ 0, cp > 0.
    #
    # This is by construction. pitch=0 means we're looking out horizontally. If
    # we look straight down, then pitch ~ 90deg. If we look past 90deg, we'll
    # keep pitch <= 90deg, but compensate by yawing by 180deg
    sp = -R_world_body[..., 2,0]
    cp = np.sqrt(1. - sp*sp)

    p = np.arctan2(sp,cp)
    y = np.arctan2(R_world_body[..., 1,0], R_world_body[..., 0,0])
    r = np.arctan2(R_world_body[..., 2,1], R_world_body[..., 2,2])

    if np.linalg.norm(R_world_cam__from_pix4d_rpy(nps.transpose(nps.cat(r,p,y)),False) - R_world_cam) > 1e-8:
        raise Exception("Inverse of R_world_cam__from_pix4d_rpy() computed incorrectly")

    return lat,lon,alt,r,p,y


def applyexif(filename,
              model_intrinsics,
              lat,lon,alt,
              # all in rad
              r,p,y,
              lens_serial_number):

    def make_Rational(x, denominator=65536):
        # It is completely ridiculous that I have to do this
        numerator = int(x*denominator)
        return fractions.Fraction(numerator,denominator)
    def make_GPSCoordinate(x):
        # It is completely ridiculous that I have to do all this
        ax = abs(x)
        d  = int(ax)
        m  = int(60*(ax - d))
        s  = 3600*(ax - d) - 60*m
        return \
            [ fractions.Fraction(d,1),
              fractions.Fraction(m,1),
              make_Rational(s) ]

    lensmodel,intrinsics = model_intrinsics.intrinsics()
    if lensmodel != 'LENSMODEL_OPENCV5':
        raise Exception("I only support opencv5 distortions")

    fxy = intrinsics[0:2]
    cxy = intrinsics[2:4]
    r1,r2,t1,t2,r3 = intrinsics[4:]


    # pixels/mm. This number is completely made up. Nothing cares about it. We
    # just need to make sure that fxy and cxy come out correctly, in the units
    # of pixels
    pixel_pitch = 1000.

    tags = { 'Xmp.Camera.Yaw':                    str(y*180./np.pi),
             # I offset the "pitch" by 90deg. From experience, this is the offset between
             # Xmp.drone-parrot.CameraPitchDegree and Xmp.Camera.Pitch. I store the
             # latter, while I computed the former. No idea why this is all like this
             'Xmp.Camera.Pitch':                  str(p*180./np.pi + 90.),
             'Xmp.Camera.Roll':                   str(r*180./np.pi),
             'Exif.GPSInfo.GPSLatitude':          make_GPSCoordinate(abs(lat)),
             'Exif.GPSInfo.GPSLongitude':         make_GPSCoordinate(abs(lon)),
             'Exif.GPSInfo.GPSAltitude':          make_Rational(alt),
             'Exif.GPSInfo.GPSLatitudeRef':       'N' if lat >= 0 else 'S',
             'Exif.GPSInfo.GPSLongitudeRef':      'E' if lon >= 0 else 'W',
             'Exif.GPSInfo.GPSAltitudeRef':       '0', # above sea level
             'Xmp.Camera.ModelType':              'perspective',
             'Xmp.Camera.PrincipalPoint':         f'{cxy[0]/pixel_pitch},{cxy[1]/pixel_pitch}',
             'Xmp.Camera.PerspectiveFocalLength': str((fxy[0] + fxy[1])/2. / pixel_pitch),
             'Xmp.Camera.PerspectiveDistortion':  ','.join(str(x) for x in (r1,r2,r3,t1,t2)),

             # I don't know how to do this, "right". From looking at other
             # people's .jpg files, I see that 3 is "cm". I'd like "mm", but I'm
             # not bothering to figure it out. This is good-enough
             'Exif.Photo.FocalPlaneResolutionUnit': 3,
             'Exif.Photo.FocalPlaneXResolution':   make_Rational(pixel_pitch * 10.),
             'Exif.Photo.FocalPlaneYResolution':   make_Rational(pixel_pitch * 10.),

             'Exif.Photo.LensSerialNumber':        lens_serial_number,
            }

    print(f"Updating {filename}")
    exif = pyexiv2.ImageMetadata(filename)
    exif.read()

    # Required to be able to touch Xmp.Camera tags
    try:
        pyexiv2.register_namespace('https://support.pix4d.com/hc/en-us/articles/205732309-EXIF-and-XMP-tag-information-read-by-Pix4D-Desktop/', 'Camera')
    except:
        pass

    for key in tags:
        if key in exif:
            print(f"WARNING: key '{key}' already exists in the metadata for '{filename}'. Modifying existing tags is an untested case, so I do nothing for now")
            continue

        if   re.match('Xmp',  key): tag = pyexiv2.XmpTag (key, tags[key])
        elif re.match('Exif', key): tag = pyexiv2.ExifTag(key, tags[key])
        else: raise Exception("I only know how to make EXIF and XMP tags")
        exif[key] = tag

    exif.write()




posefile           = '~/jpl/uavsar/2021-05--grand-canyon/batch1/gps_batch1.txt'
imageglob          = '~/jpl/uavsar/2021-05--grand-canyon/batch1/cam330068/*.jpg'
model_filename     = '~/jpl/sfmviz/cam68-dimacal-equalfxy.cameramodel'
first_image        = 0
Nimages            = 3
lens_serial_number = 'camera330068'


model_intrinsics = mrcal.cameramodel(os.path.expanduser(model_filename))
filenames, rpydeg_latlonalt_date = parse_posefile(posefile,
                                                  imageglob,
                                                  first_image,
                                                  Nimages)

proj_context__init( # lat,lon
                    (rpydeg_latlonalt_date[0,3], rpydeg_latlonalt_date[0,4] ))


# Pose correction from sfmviz feature-matching and solving
Rt_cam_camorig = np.array([[ 9.99624187e-01, -2.01710718e-03,  2.73388973e-02],
                           [ 1.49423622e-03,  9.99815841e-01,  1.91324850e-02],
                           [-2.73724548e-02, -1.90844440e-02,  9.99443111e-01],
                           [-8.61578483e+02,  1.85356373e+03,  1.09907969e+03]])


# shape (Ncam, 4,3)
Rt_world_cam = \
    Rt_world_cam__compute(rpydeg_latlonalt_date, True)

# Apply pose correction
Rt_world_cam = mrcal.compose_Rt( Rt_world_cam,
                                 mrcal.invert_Rt(Rt_cam_camorig))

# deg,deg,m,rad,rad,rad
lat,lon,alt,r,p,y = decompose_pose(Rt_world_cam)

for i in range(len(filenames)):

    applyexif(filenames[i],
              model_intrinsics,
              lat[i],lon[i],alt[i],
              r[i],p[i],y[i],
              lens_serial_number)
