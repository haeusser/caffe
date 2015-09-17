#!/usr/bin/python

from CaffeAdapter import *

def writeImage(net, blob, folder='output', suffix=None, scale=1.0):
    pass

def writeFlow(net, blob, folder='output', suffix=None, scale=1.0):
    pass

def writeFloat(net, blob, folder='output', suffix=None, scale=1.0):
    pass

def imageToRange01(net, image_blob)
    # this should return the scaled blob (nout=1)
    # proto: layer { name: "img0_scaled" type: "Eltwise" bottom: "img0" top: "img0_scaled" eltwise_param { operation: SUM coeff: 0.003921569 } }
    pass

def imageToRange0255(net, image_blob):
    # this should return the scaled blob (nout=1)
    # proto: layer { name: "img0_scaled" type: "Eltwise" bottom: "img0" top: "img0_scaled" eltwise_param { operation: SUM coeff: 255 } }
    pass

def subtractMean(net, image_blob, color, input_scale=1.0, mean_scale=1.0, ouptut_scale=1.0):
    # color = (r,g,b)
    # this should return the scaled blob (nout=1)
    # proto layer { name: "scaled_img1" bottom: "img1_ds" top: "scaled_img1" type: "Mean" mean_param: { operation: SUBTRACT input_scale: 1.0 mean_scale: 1.0 output_scale: 0.00392156862745 value: 76.4783107737 value: 69.4660111681 value: 58.0279756163 } }
    pass

def addMean(net, image_blob, color, input_scale=1.0, mean_scale=1.0, ouptut_scale=1.0):
    # color = (r,g,b)
    # this should return the scaled blob (nout=1)
    # proto layer { name: "scaled_img1" bottom: "img1_ds" top: "scaled_img1" type: "Mean" mean_param: { operation: ADD input_scale: 1.0 mean_scale: 1.0 output_scale: 0.00392156862745 value: 76.4783107737 value: 69.4660111681 value: 58.0279756163 } }
    pass

def concat(net, *args):
    # this should concat all blobs given in args
    # and should return the concatenated blob (nout=1)
    pass

def downsample(net, input, width=None, height=None, reference=None):
    # this should return the downsampled blob (nout=1)
    # if reference=None:
    # layer {
    #   name: "down0"
    #   type: "Resample"
    #   top: "img1_ds"
    #   bottom: "img1"
    #   resample_param {
    #     width: ..
    #     height: ..
    #   }
    # }
    # else:
    # layer {
    #   name: "down0"
    #   type: "Resample"
    #   top: "img1_ds"
    #   bottom: "img1"
    #   bottom: reference
    # }
