#!/usr/bin/python

import os

#os.environ['GLOG_logtostderr'] = '0'

import sys
import CNN
from PyCNN import PyCNN
import argparse
import numpy as np
import Toolbox as tb
from scipy import misc
import caffe
import cv2

pycnn=PyCNN()
pycnn.initEnvironment()

parser = argparse.ArgumentParser()
parser.add_argument('--iter', help='iteration', default=-1, type=int)
parser.add_argument('--verbose', help='verbose', action='store_true')

args = parser.parse_args()
tb.verbose = args.verbose

(modelFile, iter) = pycnn.getModelFile(args.iter)
print 'using model from iteration %d' % iter

dataset_path = '/home/ilge/data/caffe/superresolution/datasets'
#image_sets = ['Set5', 'Set14']
#image_sets = ['Set5', 'Set14', 'city', 'calendar', 'foliage', 'walk', 'coco-test', 'youtube-test']
image_sets = ['youtube-test']

out_dir = 'output_SR_%d' % iter
os.system('mkdir -p %s' % out_dir)

def center_crop(img, w, h):
    width = img.shape[1]
    height = img.shape[0]

    crop_h = (width-w)/2
    crop_v = (height-h)/2

    img=img[crop_v:(-crop_v if crop_v > 0 else None), crop_h:(-crop_h if crop_h > 0 else None), :]
    return img


def process(set,ent,inputEnt):
    highres = misc.imread(ent, flatten=False).astype(np.float32)
    lowres = misc.imread(inputEnt, flatten=False).astype(np.float32)
    width = highres.shape[1]
    height = highres.shape[0]

    print 'processing %s (%dx%d)' % (ent, width, height)

    defFile = 'scratch/test_SR_deploy.prototxt'
    pycnn.preprocessFile('deploy.prototmp', defFile, {'WIDTH': width, 'HEIGHT': height})

    if 'youtube' in set:
        print 'using youtube mean'
        mean_bgr = tb.readFloat("/misc/lmbraid17/ilge/caffe/superresolution/datasets/youtube/test/mean3.float3").astype(np.float32)
    else:
        mean_bgr = tb.readFloat("/home/ilge/data/caffe/superresolution/datasets/coco/mean.float3").astype(np.float32)

    mean_bgr = cv2.resize(mean_bgr, (width, height), interpolation=cv2.INTER_CUBIC)
    mean_bgr_lowres = cv2.resize(mean_bgr, (width/4, height/4), interpolation=cv2.INTER_CUBIC)

    highres_nomean_bgr = highres[:, :, (2, 1, 0)] - mean_bgr
    lowres_nomean_bgr = lowres[:, :, (2, 1, 0)] - mean_bgr_lowres

    caffe.set_phase_test()
    caffe.set_mode_gpu()
    caffe.set_logging_disabled()
    net = caffe.Net(
       defFile,
       modelFile
    )

    print 'network forward pass'
    blobs = net.forward(highres=np.asarray([net.preprocess('highres', highres_nomean_bgr / 255.0)]),lowres=np.asarray([net.preprocess('lowres', lowres_nomean_bgr / 255.0)]))

    output_bgr = 255.0 * blobs['output'].transpose(0, 2, 3, 1).squeeze()
    output_bgr += mean_bgr
    output_bgr[output_bgr < 0] = 0
    output_bgr[output_bgr > 255] = 255

    os.system('mkdir -p %s/%s' % (out_dir, set))
    basename = os.path.basename(ent)[:-4].replace('_GT', '')
    misc.imsave('%s/%s/%s-gt.png' % (out_dir, set, basename), highres)
    misc.imsave('%s/%s/%s-recon.png' % (out_dir, set, basename), output_bgr[:, :, (2, 1, 0)])

    #nn, li, cu = tb.computeBasePSNRs(ent, downsampledFilename=inputEnt)
    nn = tb.PSNR(); li=tb.PSNR(); cu=tb.PSNR()

    psnr = tb.PSNR()
    psnr.set(blobs['psnr'][0, 0, 0, 0],  blobs['psnr_y'][0, 0, 0, 0])

    print 'nn=%5s, li=%5s, cu=%5s, net=%5s' % (nn, li, cu, psnr)

    return (nn, li, cu, psnr)


results = {}
for set in image_sets:
    list = tb.readTupleList('%s/%s-list.txt' % (dataset_path, set))

    nn_list = tb.PSNRList()
    li_list = tb.PSNRList()
    cu_list = tb.PSNRList()
    net_list = tb.PSNRList()
    for ent in list:
        print ''

        filename = ent[0]
        if 'comic' in filename or 'ppt3' in filename or 'zebra' in filename:
            tb.notice('skipping %s' % filename,'del')
            continue

        inputFilename = filename.replace('.ppm','.caffe.downsampled.ppm')
        #if len(ent)==2:
        #    down

        (nn, li, cu, psnr) = process(set, filename, inputFilename)
        nn_list.append(nn)
        li_list.append(li)
        cu_list.append(cu)
        net_list.append(psnr)

    results[set] = (nn_list, li_list, cu_list, net_list)
    print '%15s: nn=%5s, li=%5s, cu=%5s, net=%5s' % (set, results[set][0], results[set][1], results[set][2], results[set][3])

print ''
for set in image_sets:
    print '%15s: nn=%5s, li=%5s, cu=%5s, net=%5s' % (set, results[set][0], results[set][1], results[set][2], results[set][3])
print ''




