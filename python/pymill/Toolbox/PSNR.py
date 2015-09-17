#!/usr/bin/python

import numpy as np
import cv2
import math
from scipy import misc

def psnr(img, gt, crop=0):
    if crop > 0:
        img = img[crop:-crop, crop:-crop, :]
        gt = gt[crop:-crop, crop:-crop, :]

    w = min(img.shape[1], gt.shape[1])
    h = min(img.shape[0], gt.shape[0])
    img = img[0:h, 0:w, :]
    gt = gt[0:h, 0:w, :]

    return 20.0 * math.log(255.0 / np.sqrt(np.sum(np.square(img.astype(np.float64) - gt.astype(np.float64)))
                                           / float(np.prod(gt.shape))), 10)

def psnrY(img, gt, crop=0):
    if crop > 0:
        img = img[crop:-crop, crop:-crop, :]
        gt = gt[crop:-crop, crop:-crop, :]

    w = min(img.shape[1], gt.shape[1])
    h = min(img.shape[0], gt.shape[0])
    img = img[0:h, 0:w, :]
    gt = gt[0:h, 0:w, :]

    imgY = (65.4810*img[:, :, 0] + 128.5530*img[:, :, 1] + 24.9660*img[:, :, 2]) / 255.0
    gtY =  (65.4810*gt[:, :, 0] +  128.5530*gt[:, :, 1] +  24.9660*gt[:, :, 2]) / 255.0

    return 20.0 * math.log(255.0 / np.sqrt(np.sum(np.square(imgY.astype(np.float64) - gtY.astype(np.float64)))
                                           / float(np.prod(gtY.shape))), 10)

class PSNR:
    def __init__(self, img=None, gt=None, crop=0):
        if img != None and gt != None:
            self._psnr = psnr(img, gt, crop)
            self._psnrY = psnrY(img, gt, crop)
        else:
            self._psnr = 0
            self._psnrY = 0

    def add(self,psnr):
        self._psnr += psnr.rgb()
        self._psnrY += psnr.y()

    def set(self, rgb, y):
        self._psnr = rgb
        self._psnrY = y

    def divide(self,x):
        self._psnr /= float(x)
        self._psnrY /= float(x)

    def rgb(self):
        return self._psnr

    def y(self):
        return self._psnrY

    def __str__(self):
        #return '%5.2f[%5.2f]' % (self.rgb(), self.y())
        return '%5.2f' % (self.rgb())

    def __repr__(self):
        return str(self)

class PSNRList:
    def __init__(self):
        self._list = []

    def append(self, psnr=None, img=None, gt=None, crop=0):
        if psnr != None:
            self._list.append(psnr)
        else:
            self._list.append(PSNR(img, gt, crop))

    def mean(self):
        psnr = PSNR()
        for ent in self._list:
            psnr.add(ent)
        psnr.divide(len(self._list))
        return psnr

    def __str__(self):
        return str(self.mean())

    def __repr__(self):
        return str(self)

def computeBasePSNRs(ent,downsampledExt='.caffe.downsampled.ppm',downsampledFilename=None, crop=4):
    img = misc.imread(ent, flatten=False)
    width = img.shape[1]
    height = img.shape[0]

    if downsampledFilename == None:
        downsampledFilename = ent.replace('.ppm', downsampledExt)
    img_ds = misc.imread(downsampledFilename)

    nn = PSNR(img=cv2.resize(img_ds, (width, height), interpolation=cv2.INTER_NEAREST), gt=img, crop=crop)
    li = PSNR(img=cv2.resize(img_ds, (width, height), interpolation=cv2.INTER_LINEAR), gt=img, crop=crop)
    cu = PSNR(img=cv2.resize(img_ds, (width, height), interpolation=cv2.INTER_CUBIC), gt=img, crop=crop)

    #misc.imsave(ent.replace('.ppm', '.cv2.upsampled.bicubic.ppm'), cv2.resize(img_ds, (width, height), interpolation=cv2.INTER_CUBIC))

    return (nn, li, cu)
