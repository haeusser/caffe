#!/usr/bin/python

from CaffeAdapter import *
import pymill.Toolbox as tb

def readImage(net, filename, num=1):
    return Layers.ImgReader(net,
                            nout=1,
                            reader_param = {
                                'file': filename,
                                'num': num
                            })

Network.readImage = readImage

def readFloat(net, filename, num=1):
    return Layers.FloatReader(net,
                            nout=1,
                            reader_param = {
                                'file': filename,
                                'num': num
                            })

Network.readFloat = readFloat

def writeImage(net, blob, folder='output', prefix='image', suffix='', scale=1.0, filename=None):
    if filename is not None:
        Layers.ImgWriter(net,
                         blob,
                         writer_param={
                           'file' : filename,
                           'scale' : scale })
    else:
        Layers.ImgWriter(net,
                         blob,
                         writer_param={
                           'folder' : folder,
                           'prefix': prefix,
                           'suffix': suffix,
                           'scale' : scale })

Network.writeImage = writeImage

def writeFlow( net, blob, folder='output', prefix='flow', suffix='', scale=1.0, filename=None):
    if filename is not None:
        Layers.FLOWriter(net,
                         blob,
                          writer_param={
                            'file' : filename,
                            'scale' : scale })
    else:
        Layers.FLOWriter(net,
                         blob,
                          writer_param={
                            'folder' : folder,
                            'prefix': prefix,
                            'suffix': suffix,
                            'scale' : scale })

Network.writeFlow = writeFlow

def writeFloat(net, blob, folder='output', prefix='', suffix='', scale=1.0, filename=None):
    if filename is not None:
        Layers.FloatWriter(net,
                           blob,
                           writer_param={
                             'file' : filename,
                             'prefix': prefix,
                             'suffix': suffix,
                             'scale' : scale })
    else:
        Layers.FloatWriter(net,
                           blob,
                           writer_param={
                             'folder' : folder,
                             'prefix': prefix,
                             'suffix': suffix,
                             'scale' : scale })

Network.writeFloat = writeFloat

def scale(net, image_blob, factor):
    return Layers.Eltwise(net,
                          image_blob,
                          nout=1,
                          eltwise_param={
                            'operation': Params.Eltwise.SUM,
                            'coeff': (factor,)
                          })

Network.scale = scale

def imageToRange01(net, image_blob):
    return Layers.Eltwise(net,
                          image_blob,
                          nout=1,
                          eltwise_param={
                            'operation': Params.Eltwise.SUM,
                            'coeff': (1.0/255.0,)
                          })

Network.imageToRange01 = imageToRange01

def imageToRange0255(net, image_blob):
    return Layers.Eltwise(net,
                          image_blob,
                          nout=1,
                          eltwise_param={
                            'operation': Params.Eltwise.SUM,
                            'coeff': (255.0,)
                          })

Network.imageToRange0255 = imageToRange0255

def subtractMean(net, image_blob, color, input_scale=1.0, mean_scale=1.0, ouptut_scale=1.0):
    return Layers.Mean(net,
                       image_blob,
                       nout=1,
                       mean_param={
                         'operation': Params.Mean.SUBTRACT,
                         'input_scale': input_scale,
                         'mean_scale': mean_scale,
                         'output_scale': output_scale,
                         'value': color
                       })

Network.subtractMean = subtractMean

def addMean(net, image_blob, color, input_scale=1.0, mean_scale=1.0, ouptut_scale=1.0):
    return Layers.Mean(net,
                       image_blob,
                       nout=1,
                       mean_param={
                         'operation': Params.Mean.ADD,
                         'input_scale': input_scale,
                         'mean_scale': mean_scale,
                         'output_scale': output_scale,
                         'value': color
                       })

Network.addMean = addMean

def concat(net, *args):
    '''
    @brief Setup a ConcatLayer that takes all ARGS and throws them together along the first dimension
    @param net Current network
    @returns A new blob (concatenation of all blobs in ARGS)
    '''
    return Layers.Concat(net,
                         args,
                         nout=1,
                         concat_param={'concat_dim': 1})

Network.concat = concat

def resample(net, input, width=None, height=None, reference=None, type='LINEAR', antialias=True):
    #if type=='LINEAR': type=Params.Resample.LINEAR
    #elif type=='CUBIC': type=Params.Resample.CUBIC

    if reference is not None:
        return Layers.Resample(net,
                               (input, reference),
                               nout=1,
                               resample_param={
                                 'antialias': antialias,
                                 'type': getattr(Params.Resample, type)
                               })
    else:
        return Layers.Resample(net,
                               input,
                               nout=1,
                               resample_param={
                                 'width': width,
                                 'height': height,
                                 'antialias': antialias,
                                 'type': getattr(Params.Resample, type)
                               })

Network.resample = resample

def UniformBernoulli(mean, spread, exp=False, prob=1.0):
    return {
      'rand_type': "uniform_bernoulli",
      'mean': mean,
      'spread': spread,
      'exp': exp,
      'prob': prob,
    }

def GaussianBernoulli(mean, spread, exp=False, prob=1.0):
    return {
      'rand_type': "gaussian_bernoulli",
      'mean': mean,
      'spread': spread,
      'exp': exp,
      'prob': prob,
    }

