#!/usr/bin/python3

import sys
import numpy as np
import numpysane as nps
import vnlog
import glob
import mrcal
import re
import json
import os
import os.path


metadata_glob   = '/mnt/aargh/proj/uavsar/2021-06-21-23-quakes-mt-whitney/2021-07-23--processed/batch1/*.vnl'
images_dir      = '/mnt/diskA/2026-03--quakes/whitney'
camera_ids      = ('SN330071', # the first one is the origin of the rig
                   'SN330075')
project_dir     = '/mnt/diskA/2026-03--quakes/opensfm-whitney-smalldataset-rig-opk'



# adapted from OpenSfM/opensfm/geometry.py
# THIS function broadcasts fully
def R_from_opk(opk):
    o = np.array(opk)
    p = np.array(opk)
    k = np.array(opk)

    o[...,0 ] *= -1
    o[...,1:] = 0

    p[...,0] = 0
    p[...,1] *= -1
    p[...,2] = 0

    k[...,:2] = 0
    k[...,2 ] *= -1


    Rw = mrcal.R_from_r(o)
    Rp = mrcal.R_from_r(p)
    Rk = mrcal.R_from_r(k)

    Rc = np.array([[1,  0, 0],
                   [0, -1, 0],
                   [0,  0, -1]])
    return nps.matmult(Rc, Rk, Rp, Rw)

def r_from_opk(opk):
    return mrcal.r_from_R(R_from_opk(opk))





# adapted from OpenSfM/opensfm/geo.py
# THIS function broadcasts fully
def ecef_from_lla(lla):
    WGS84_a = 6378137.0
    WGS84_b = 6356752.314245
    a2 = WGS84_a*WGS84_a
    b2 = WGS84_b*WGS84_b
    lat = np.radians(lla[...,0])
    lon = np.radians(lla[...,1])
    clat = np.cos(lat)
    slat = np.sin(lat)
    L = 1.0 / np.sqrt(a2 * clat*clat + b2 * slat*slat)
    return nps.glue( nps.dummy( (a2 * L + lla[...,2]) * clat * np.cos(lon), axis=-1),
                     nps.dummy( (a2 * L + lla[...,2]) * clat * np.sin(lon), axis=-1),
                     nps.dummy( (b2 * L + lla[...,2]) * slat,               axis=-1),
                     axis = -1 )
def topocentric_from_ecef_transform(lla):
    xyz = ecef_from_lla(lla)
    sa = np.sin(np.radians(lla[...,0]))
    ca = np.cos(np.radians(lla[...,0]))
    so = np.sin(np.radians(lla[...,1]))
    co = np.cos(np.radians(lla[...,1]))
    # A = np.array(( (-so,    -sa*co, ca*co),
    #                (co,     -sa*so, ca*so),
    #                (0,      ca,     sa),    ))
    Ainv =     np.array(( ( -so,   co,     0),
                          (-co*sa, -sa*so, ca ),
                          ( ca*co,  ca*so, sa ),))
    # inverse; y = A x + b -> x = Ainv y - Ainv b
    return ( Ainv, -nps.inner(Ainv,xyz ) )



def process_metadata(metadata_glob,
                     camera_ids,
                     # I only report images that exist in this directory
                     images_dir):

    regex_use_image = "|".join(camera_ids) ####### error-prone
    dtype = np.dtype([ ('filename', 'U256'),
                       ('latitude longitude altitude_m omega_deg phi_deg kappa_deg', float, (6,)),])

    images_from_time_id = dict()
    rig_cameras         = dict()
    exif_overrides      = dict()

    ######## for visualization; currently in viz-opk.py; move here?
    lla_opk = np.zeros( (0,6), dtype=float)
    images  = np.zeros( (0,),  dtype='U256')

    tmin = 1e20
    for f in glob.glob(metadata_glob):

        arr = vnlog.slurp(f, dtype=dtype)
        N = len(arr)
        mask_keep = np.zeros((N,), dtype=bool)
        paths = arr['filename']
        filenames = np.array(paths)
        times = np.zeros((N,), dtype=float)
        ids   = np.zeros((N,), dtype='U32')

        for i_image in range(N):
            i = paths[i_image].rindex('/')
            filenames[i_image] = paths[i_image][i+1:]
            if re.search(regex_use_image, filenames[i_image]) and \
               os.path.exists( f"{images_dir}/{filenames[i_image]}" ):
                mask_keep[i_image] = True

                m = re.match('[0-9]+_[0-9]+_([0-9]+)_([0-9]+)_(SN[0-9]+)', filenames[i_image])
                times[i_image] = float(m.group(1)) + float(m.group(2))/1000.
                ids[i_image] = m.group(3)

        lla_opk_here   = arr['latitude longitude altitude_m omega_deg phi_deg kappa_deg'][mask_keep]
        filenames_here = filenames[mask_keep]
        times_here     = times[mask_keep]
        ids_here       = ids[mask_keep]

        lla_opk_here[..., 3:] *= np.pi/180.

        lla_opk = nps.glue( lla_opk,
                            lla_opk_here,
                            axis = -2 )
        images = nps.glue( images,
                           filenames_here,
                           axis = -1 )

        for i in range(len(times_here)):
            t        = times_here[i]
            ID       = str(ids_here[i])
            filename = filenames_here[i]

            if t not in images_from_time_id:
                images_from_time_id[t] = dict()
            images_from_id = images_from_time_id[t]
            if ID in images_from_id:
                raise Exception(f"Camera {ID} at t={t} appears more than once")
            images_from_id[ID] = dict(lla_opk = lla_opk_here[i],
                                      filename = filename)

        if len(times_here):
            tmin = min(tmin, np.min(times_here))

    for (t,images_from_id) in images_from_time_id.items():
        if not all(k in images_from_id for k in camera_ids):
            continue
        # This time instance has images from ALL the cameras I care about
        camera_ref = camera_ids[0]
        r_ref_cam0 = r_from_opk(images_from_id[camera_ref]['lla_opk'][3:])
        for k in camera_ids:
            r_ref_cam = r_from_opk(images_from_id[k]['lla_opk'][3:])
            r_cam_cam0 = mrcal.compose_r(-r_ref_cam, r_ref_cam0)

            rig_cameras[k] = dict(rotation    = list(r_cam_cam0),
                                  translation = (0,0,0),
                                  camera      = k)
        break
    else:
        raise Exception("No instances in time had data from ALL my cameras")


    def generate_rig_snapshot(t,images_from_id):
        return \
            [ (d['filename'], ID) \
              for (ID,d) in images_from_id.items()]
    rig_assignments = [ generate_rig_snapshot(t,images_from_id) \
                        for (t,images_from_id) in images_from_time_id.items() ]

    reference_lla = None
    for (t,images_from_id) in images_from_time_id.items():
        for (ID,d) in images_from_id.items():
            exif_overrides[d['filename']] = \
                dict(gps = dict(latitude  = d['lla_opk'][0],
                                longitude = d['lla_opk'][1],
                                altitude  = d['lla_opk'][2]),
                     opk = dict(omega     = d['lla_opk'][3] * 180./np.pi,
                                phi       = d['lla_opk'][4] * 180./np.pi,
                                kappa     = d['lla_opk'][5] * 180./np.pi),
                     camera = ID)

            if t == tmin:
                # There might be more than one of these; that should be ok: the
                # position should be identical
                reference_lla = d['lla_opk'][:3]
                reference_lla[2] = 0 # altitude=0

    if reference_lla is None:
        raise Exception("No reference_lla found. This is a bug")
    reconstruction_initial = [dict(cameras       = dict(),
                                   shots         = dict(),
                                   points        = dict(),
                                   reference_lla = dict(latitude  = reference_lla[0],
                                                        longitude = reference_lla[1],
                                                        altitude  = reference_lla[2]))]
    # Made-up. Can be anything
    for ID in camera_ids:
        reconstruction_initial[0]['cameras'][ID] = \
            dict(projection_type = "perspective",
                 width           = 640,
                 height          = 480,
                 focal           = 15,
                 k1              = 0,
                 k2              = 0)
    # Made-up. Can be anything. No points at all makes the opensfm viewer fail
    reconstruction_initial[0]['points'] = {"0": dict(color       = (0,0,0),
                                                     coordinates = (0,0,0))}

    Aref,xyzref = topocentric_from_ecef_transform(reference_lla)
    for (t,images_from_id) in images_from_time_id.items():
        for (ID,d) in images_from_id.items():
            rt_enu_cam = nps.glue( r_from_opk(d['lla_opk'][3:]),
                                   nps.inner(Aref, ecef_from_lla(d['lla_opk'][:3])) + xyzref,
                                   axis = -1 )
            rt_cam_enu = mrcal.invert_rt(rt_enu_cam)
            reconstruction_initial[0]['shots'][d['filename']] = \
                dict(rotation     = list(rt_cam_enu[:3]),
                     translation  = list(rt_cam_enu[3:]),
                     camera       = ID,
                     gps_position = list(rt_enu_cam[3:]),

                     # These are made-up. Without these the opensfm viewer fails
                     gps_dop      = 15,
                     orientation  = 1,
                     capture_time = 0,
                     vertices     = [],
                     faces        = [],
                     scale        = 1.0,
                     covariance   = [],
                     merge_cc     = 0)


    return \
        ( rig_cameras,
          rig_assignments,
          exif_overrides,
          reconstruction_initial )

def symlink_images(project_dir,
                   images_dir,
                   filenames):

    try: os.mkdir(project_dir)
    except FileExistsError: pass

    os.mkdir(f"{project_dir}/images")

    for f in filenames:
        os.symlink( f"{images_dir}/{f}",
                    f"{project_dir}/images/{f}" )




( rig_cameras,
  rig_assignments,
  exif_overrides,
  reconstruction_initial ) = process_metadata(metadata_glob,
                                              camera_ids,
                                              images_dir)

symlink_images(project_dir,
               images_dir,
               exif_overrides.keys())

with open(f"{project_dir}/rig_cameras.json", "w") as f:
    json.dump(rig_cameras, f)
with open(f"{project_dir}/rig_assignments.json", "w") as f:
    json.dump(rig_assignments, f)
with open(f"{project_dir}/exif_overrides.json", "w") as f:
    json.dump(exif_overrides, f)
with open(f"{project_dir}/reconstruction_initial.json", "w") as f:
    json.dump(reconstruction_initial, f)
