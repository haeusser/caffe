import os
import sys

__author__ = 'fischer'

def makeListStereo(basepath, imin, imax):
    str = ''
    for idx in range(imin, imax+1):
        file_l = os.path.join(basepath, 'image_2/%06d_10.png' % idx)
        file_r = os.path.join(basepath, 'image_3/%06d_10.png' % idx)
        if not os.path.isfile(file_l):
            print("Not existant: " + file_l)
            sys.exit(1)

        if not os.path.isfile(file_r):
            print("Not existant: " + file_r)
            sys.exit(1)
        str += '%s %s\n' % (file_l, file_r)

    return str


outfolder = '/misc/lmbraid17/sceneflownet/common/data/3_other-ds/kitti/'

# KITTI 2012
str = makeListStereo('/misc/lmbraid17/sceneflownet/common/data/3_other-ds/kitti/2012/original/testing/',
               0, 194)

with open(os.path.join(outfolder, "kitti2012_test.txt"), "w") as text_file:
    text_file.write(str)

# KITTI 2015
str = makeListStereo('/misc/lmbraid17/sceneflownet/common/data/3_other-ds/kitti/2015/original_sceneflow/testing/',
               0, 199)

with open(os.path.join(outfolder, "kitti2015_test.txt"), "w") as text_file:
    text_file.write(str)
