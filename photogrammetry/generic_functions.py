r'''Pose-parsing functions used by raw-convert.py

raw-convert.py imports these from generic_functions.py in sfmviz
(https://github.jpl.nasa.gov/kogan/sfmviz), which isn't publicly available. The
coordinate-system and rotation code here is copied verbatim from raw-convert.py
as of fa4d0ba, the last commit before it was moved to sfmviz (3d5cab7).
parse_filenames_poses_from_metadata() is reconstructed from that code, reading
the "images-referenced" vnl format written by reference-images-ins:

  # time serialnumber filename latitude longitude altitude_ft roll_deg pitch_deg heading_deg

'''

import os
import numpy as np
import numpysane as nps
import pyproj

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

def compute__R_world_cam(rpy__plane,
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

We return a rotation to map mrcal-style camera coords (right,down,forward) to
world coords (north,east,down)

    '''

    r,p,y = nps.transpose(rpy__plane)
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

def compute__Rt_world_cam(rpy_latlonalt__plane,
                          camera_roll_deg       = 0,
                          camera_yaw_deg        = 0,
                          camera_pitch_deg      = 0,
                          camera_pitch_then_yaw = False):

    r'''Computes the world <-> cam transformation

Returns a transformation (in an (Ncameras,4,3) array) to map mrcal-style camera
coords (right,down,forward) to world coords (north,east,down).

    '''

    # shape (...,3,3)
    R_world_cam = compute__R_world_cam(rpy_latlonalt__plane[:,:3],
                                       camera_roll_deg       = camera_roll_deg,
                                       camera_yaw_deg        = camera_yaw_deg,
                                       camera_pitch_deg      = camera_pitch_deg,
                                       camera_pitch_then_yaw = camera_pitch_then_yaw)

    lat,lon,alt = nps.transpose(rpy_latlonalt__plane[:, 3:6])

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

def parse_filenames_poses_from_metadata(metadata,
                                        camera_roll_deg       = 0,
                                        camera_yaw_deg        = 0,
                                        camera_pitch_deg      = 0,
                                        camera_pitch_then_yaw = False,
                                        image_path_prefix     = None,
                                        image_directory       = None):
    r'''Reads image filenames and airplane poses from a metadata vnl

Returns (filenames, Rt_world_cam). Rt_world_cam has shape (N,4,3), and maps
mrcal-style camera coords (right,down,forward) to world coords
(north,east,down). The local world coordinate system is centered at the first
image

    '''

    if image_path_prefix is not None and image_directory is not None:
        raise Exception("image_path_prefix and image_directory are mutually exclusive")

    # This has a string column, so I parse it by hand: older vnlog.slurp()
    # can't read non-numerical data
    list_keys = None
    rows      = []
    with open(metadata) as f:
        for l in f:
            if l.startswith('##'):
                continue
            if l.startswith('#'):
                if list_keys is None:
                    list_keys = l[1:].split()
                continue
            fields = l.split()
            if len(fields) == 0:
                continue
            if list_keys is None or len(fields) != len(list_keys):
                raise Exception(f"Couldn't parse line in '{metadata}': '{l.strip()}'")
            rows.append(fields)
    if len(rows) == 0:
        raise Exception(f"No data in '{metadata}'")
    x = np.array(rows, dtype=str)
    dict_key_index = { k: i for i,k in enumerate(list_keys) }

    for k in ('filename','latitude','longitude','altitude_ft',
              'roll_deg','pitch_deg','heading_deg'):
        if k not in dict_key_index:
            raise Exception(f"Metadata '{metadata}' is missing the '{k}' column")

    filenames = x[:, dict_key_index['filename']]
    if image_path_prefix is not None:
        filenames = [f"{image_path_prefix}/{f}" for f in filenames]
    elif image_directory is not None:
        filenames = [f"{image_directory}/{os.path.basename(f)}" for f in filenames]
    else:
        filenames = list(filenames)

    colidx = (dict_key_index['roll_deg'],
              dict_key_index['pitch_deg'],
              dict_key_index['heading_deg'],
              dict_key_index['latitude'],
              dict_key_index['longitude'],
              dict_key_index['altitude_ft'] )
    rpy_latlonalt__plane = x[:, colidx].astype(float)

    # altitude: ft -> m
    rpy_latlonalt__plane[:,5] *= 12.*2.54/100.
    rpy_latlonalt__plane[:,:3] *= np.pi/180.

    proj_context__init( # lat,lon
                        (rpy_latlonalt__plane[0,3], rpy_latlonalt__plane[0,4] ),
                        allow_reinit = True)

    Rt_world_cam = \
        compute__Rt_world_cam(rpy_latlonalt__plane,
                              camera_roll_deg       = camera_roll_deg,
                              camera_yaw_deg        = camera_yaw_deg,
                              camera_pitch_deg      = camera_pitch_deg,
                              camera_pitch_then_yaw = camera_pitch_then_yaw)
    return filenames, Rt_world_cam


def parse_filenames_poses_from_exif(images, *args, **kwargs):
    raise Exception("Reading poses from EXIF isn't available in this copy of generic_functions.py; pass --metadata instead")
