#!/usr/bin/python3
# Make an OpenSfM rig reconstruction readable by ODM's bundled (older) OpenSfM.
# ODM's export_geocoords moves every shot with shot.pose.set_rotation_matrix(); a shot that
# belongs to a multi-camera rig instance has an immutable pose there ("Cannot set an
# immutable pose"). Every shot in reconstruction.json already stores its world pose, so
# dropping rig_cameras and rig_instances turns each shot into its own trivial rig with the
# same pose. The rig reconstruction is kept as reconstruction.rig.json.
#   flatten-rig.py PROJECT/opensfm
import sys, os, json, numpy as np

def rotation_matrix(r):
    r = np.asarray(r, float); th = np.linalg.norm(r)
    if th < 1e-12: return np.eye(3)
    k = r / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K

d = sys.argv[1]
src, rig = f'{d}/reconstruction.json', f'{d}/reconstruction.rig.json'
if not os.path.exists(rig):
    os.rename(src, rig)
recs = json.load(open(rig))
for r in recs:
    if 'rig_instances' not in r: continue
    # check: shot pose == rig camera pose o rig instance pose (world -> camera)
    err = 0
    for ri in r['rig_instances'].values():
        Ri, ti = rotation_matrix(ri['rotation']), np.array(ri['translation'])
        for s, c in ri['rig_camera_ids'].items():
            rc = r['rig_cameras'][c]; Rc, tc = rotation_matrix(rc['rotation']), np.array(rc['translation'])
            sh = r['shots'][s]
            err = max(err, np.abs(Rc @ Ri - rotation_matrix(sh['rotation'])).max(),
                      np.abs(Rc @ ti + tc - np.array(sh['translation'])).max())
    if err > 1e-6:
        sys.exit(f'flatten-rig: shot poses differ from rig poses by {err:g}; not flattening')
    print(f'flatten-rig: {len(r["shots"])} shots, {len(r["rig_instances"])} rig instances, '
          f'{len(r["rig_cameras"])} rig cameras -> one trivial rig per shot (max pose mismatch {err:.1e})')
    del r['rig_instances'], r['rig_cameras']
json.dump(recs, open(src, 'w'))
