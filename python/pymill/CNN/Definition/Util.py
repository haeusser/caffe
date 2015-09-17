#!/usr/bin/python

from CaffeAdapter import *

def readImage(net, filename, num=1):
    return Layers.ImageData(net, source=filename, batch_size=num)

def readFloat(net, filename, num=1):
    return Layers.FloatReader(net, file=filename, num=num)

def writeImage(net, blob, folder='output', prefix='image', suffix='', scale=1.0):
    Layers.ImgWriter(net,
                     blob,
                     writer_param={
                       'folder' : folder,
                       'prefix': prefix,
                       'suffix': suffix,
                       'scale' : scale })

def writeFlow( net, blob, folder='output', prefix='flow', suffix='', scale=1.0):
    Layers.FLOWriter(net,
                             blob,
                              writer_param={
                                'folder' : folder,
                                'prefix': prefix,
                                'suffix': suffix,
                                'scale' : scale })

def writeFloat(net, blob, folder='output', prefix='', suffix='', scale=1.0):
    Layers.FloatWriter(net,
                       blob,
                       writer_param={
                         'folder' : folder,
                         'prefix': prefix,
                         'suffix': suffix,
                         'scale' : scale })

def imageToRange01(net, image_blob):
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
    '''
    @brief Setup a ConcatLayer that takes all ARGS and throws them together along the first dimension
    @param net Current network
    @returns A new blob (concatenation of all blobs in ARGS)
    '''
    # this should concat all blobs given in args
    # and should return the concatenated blob (nout=1)
    return Layers.Concat(net,
                         args,
                         nout=1,
                         concat_param={'concat_dim': 1})

def resample(net, input, width=None, height=None, reference=None, antialias=False):
#    if reference:

#    return Layers.Resample(net, input, )
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
    pass
