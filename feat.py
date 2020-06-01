#!/usr/bin/python2

import numpy      as np
import numpysane  as nps
import scipy.spatial
import subprocess
import cv2
import sys
import re





threshold = 40







def addsuffix(filename, suffix):
    '''{xxx}.jpg -> {xxx}suffix'''
    m = re.match("(.*)\....$", filename)
    if m is not None:
        return m.group(1) + suffix
    return filename + suffix

def get_homography(images, context = None, distance_threshold = 80):
    r'''Compute a global homography between two images

    This is somewhat hacky, and needs more error detection and diagnostics. I
    use OpenCV to compute rotation-invariant features, match them up, and try to
    robustly compute a homography between the two of them

    '''

    # create computational objects, if necessary. ORB with WTA_K > 2 REQUIRES
    # cv2.NORM_HAMMING2. The default is WTA_K==2, which uses cv2.NORM_HAMMING
    # instead. WTA_K==4 works a bit better maybe
    if context is None: context = {}
    if 'detector' not in context:
        context['detector'] = cv2.ORB_create(WTA_K=4)
    if 'matcher' not in context:
        context['matcher'] = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)

    keypoint_descriptors = [ context['detector'].detectAndCompute(image,None) for image in images ]
    keypoints            = [ kp[0] for kp in keypoint_descriptors ]
    descriptors          = [ kp[1] for kp in keypoint_descriptors ]

    matches = context['matcher'].match(*descriptors)

    matches = [m for m in matches if m.distance < distance_threshold]

    p0 = np.array( [keypoints[0][m.queryIdx].pt for m in matches] )
    p1 = np.array( [keypoints[1][m.trainIdx].pt for m in matches] )

    s = cv2.findHomography(p0,p1, method=cv2.RANSAC, ransacReprojThreshold=30)


    # print '# x1 y1 x2 y2 distance'
    # matches = sorted(matches, key=lambda m: m.distance)
    # for m in matches:
    #     print \
    #         keypoints[0][m.queryIdx].pt[0], \
    #         keypoints[0][m.queryIdx].pt[1], \
    #         keypoints[1][m.trainIdx].pt[0], \
    #         keypoints[1][m.trainIdx].pt[1], \
    #         m.distance

    # P = nps.matmult(nps.glue(p0, np.ones((len(p0),1)),axis=-1),nps.transpose(s[0]))
    # diff = P[:,:2]/P[:,(2,)] - p1
    # sys.stderr.write("diff: {}\n".format(diff))
    # sys.stderr.write("difflen: {}\n".format(np.sqrt(nps.inner(diff,diff))))
    # sys.stderr.write("homography err: {}\n".format(np.linalg.norm(diff)))
    # sys.stderr.write("homography: {}\n".format(s[0].ravel()))

    return s[0]

def construct_featuretrack_homography(imagefilename0, imagefilename1):
    r'''Constructs an estimate homography for img_any

    The "feature_track" tool from img_any can accept a homography as a seed in
    its search. If there's significant motion between consecutive frames, this
    is an absolute requirement for feature_track to work at all. This homography
    is passed-in with "-H file" on the commandline where "file" a single line of
    ascii text

        h00 h01 h02 h10 h11 h12 h20 h21 h22 cols rows

    This is the homography followed by the dimensions of the image. I don't know
    what to do if the images have different dimensions

    '''
    images = [cv2.imread(f,0) for f in (imagefilename0,imagefilename1)]

    H,W = images[0].shape
    homography = get_homography(images)

    return ('{} ' * 9).format(*homography.ravel()) + '{} {}'.format(W,H) + '\n'

def featuretrack(imagefilename_prev, imagefilename, imagefilename_next,
                 homography_cookie=None,
                 suffix = '.vnl',
                 debug = False):
    r'''Invoke feature_track to produce corresponding features

    This works on two consecutive images'''

    cmd = 'rh7 /home/dima/src_boats/img_any/build/binsrc/feature_track '+ \
          '--dist_feature=4 '                                           + \
          '--feature_radius=5 '                                         + \
          '--scale=8 '                                                  + \
          '--search_radius=200 '                                        + \
          '-S 20 '                                                      + \
          '-M 40000 '                                                   + \
          '-n 1 '                                                       + \
          '--corr_threshold=.85 '                                       + \
          '--only-good-features '                                       + \
          '-w 8 '                                                       + \
          '-h 1 '

    postfilter = ' awk "/#ID/,0" | vnl-filter -p "ID,Feat[12]"'

    if homography_cookie is not None:
        homography_cookie_filename = addsuffix(imagefilename, '.homographycookie')
        with open(homography_cookie_filename, "w") as f:
            f.write(homography_cookie)
        cmd += ' -H ' + homography_cookie_filename

    cmd += ' {} {}'.format(imagefilename,imagefilename_next)

    if imagefilename_prev is not None:
        cmd += ' <( < {} vnl-filter -p ID,Feat2)'.format(addsuffix(imagefilename_prev, '.vnl'))

    cmd += " | " + postfilter

    if debug:
        sys.stderr.write("about to invoke '{}'\n".format(cmd))

    try:
        out = subprocess.check_output(cmd, shell=True, executable='/bin/zsh')
    except:
        sys.stderr.write("Error invoking '{}'\n".format(cmd))
        raise

    out_filename = addsuffix(imagefilename, suffix)
    with open(out_filename, "w") as f:
        f.write(out)


def join_continuations(imagefilename, threshold):

    query = np.loadtxt(addsuffix(imagefilename, '-independent.vnl'))
    mask  = np.loadtxt(addsuffix(imagefilename, '-continued.vnl'))

    mask_tree         = scipy.spatial.cKDTree(mask[:,1:3])
    distances,indices = mask_tree.query(query[:,1:3], distance_upper_bound=threshold)

    # I report on those points in the query set that have a too-high distance
    p = query[np.arange(distances.shape[0])[ np.isinf(distances) ], :]

    with open(addsuffix(imagefilename, '-new.vnl'), "w") as f:
        np.savetxt(f, p, fmt='%d %f %f %f %f', header='ID Feat1x Feat1y Feat2x Feat2y')






if len(sys.argv) < 3:
    raise Exception("Need image sequence on the cmdline")

imagefilenames = sorted(sys.argv[1:])

featuretrack(None, *imagefilenames[:2])

for i in xrange(1,len(imagefilenames)-1):
    sys.stderr.write("starting {}\n".format(imagefilenames[i]))

    homography_cookie = \
        construct_featuretrack_homography(imagefilenames[i],
                                          imagefilenames[i+1])

    featuretrack(*imagefilenames[i-1:i+2],
                 homography_cookie=homography_cookie,
                 suffix='-continued.vnl')

    featuretrack(None, *imagefilenames[i:i+2],
                 homography_cookie=homography_cookie,
                 suffix='-independent.vnl')

    join_continuations(imagefilenames[i], threshold)

    lastID = int(subprocess.check_output("tail -n1 " + \
                                         addsuffix(imagefilenames[i-1],'.vnl') + \
                                         " | awk '{print $1}'",
                                         shell=True))
    sys.stderr.write("lastID: {}\n".format(lastID))
    subprocess.check_output("cat {} > {}". \
                            format(addsuffix(imagefilenames[i], '-continued.vnl'),
                                   addsuffix(imagefilenames[i], '.vnl')),
                            shell=True)

    subprocess.check_output('< {} vnl-filter -p "ID=NR+{},Feat" >> {}'. \
                            format(addsuffix(imagefilenames[i], '-new.vnl'),
                                   lastID,
                                   addsuffix(imagefilenames[i], '.vnl')),
                            shell=True)










# #!/bin/zsh

# set -e

# function imagecode() {
#     printf "%05d" $1
# }

# function imagefilename() {
#     printf "G%07d.JPG" $1
# }

# function postfilter() {
#     awk '/#ID/,0' | vnl-filter -p 'ID,Feat[12]'
# }

# ifirst=20055
# ilast=20477

# ihere=$ifirst
# ihere_code=`imagecode $ihere`

# $FEATURE_TRACK $params `imagefilename $ihere` `imagefilename $(($ihere+1))` |
#     postfilter > ${ihere_code}.vnl

# for ((ihere=$(($ifirst+1)); ihere<$ilast; ihere++)) {
#         echo "starting $ihere"
#         ihere_code=`imagecode $ihere`
#         inext_code=`imagecode $(($ihere+1))`
#         iprev_code=`imagecode $(($ihere-1))`
#         $FEATURE_TRACK $params `imagefilename $ihere` `imagefilename $(($ihere+1))` \
#                        <( < ${iprev_code}.vnl vnl-filter -p ID,Feat2) |
#             postfilter > ${ihere_code}-continued.vnl

#         $FEATURE_TRACK $params `imagefilename $ihere` `imagefilename $(($ihere+1))` |
#             postfilter > ${ihere_code}-independent.vnl

#         python get_all_new_points.py --threshold 40 ${ihere_code}-independent.vnl ${ihere_code}-continued.vnl > ${ihere_code}-new.vnl
#         lastID=`tail -n1 ${iprev_code}.vnl | awk '{print $1}'`
#         echo "lastID: $lastID"
#         cat ${ihere_code}-continued.vnl > ${ihere_code}.vnl
#         < ${ihere_code}-new.vnl vnl-filter -p "ID=NR+$lastID,Feat" >> ${ihere_code}.vnl
#     }
