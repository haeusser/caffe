#!/usr/bin/python

from CaffeAdapter import *

class Blocks:
    @staticmethod
    def ConvolutionReLU(net, input, **kwargs):
        negative_slope = 0.1
        if 'negative_slope' in kwargs:
            negative_slope = kwargs['negative_slope']
            del kwargs['negative_slope']

        if not isinstance(input,tuple):
            conv = Layers.Convolution(net, input, **kwargs)
            Layers.ReLU(net, conv, nout=1, in_place=True, negative_slope=negative_slope)
            return conv
        else:
            convs = Layers.Convolution(net, input, **kwargs)
            for conv in convs:
                Layers.ReLU(net, conv, nout=1, in_place=True, negative_slope=negative_slope)
            return convs
        return

    @staticmethod
    def DeconvolutionReLU(net, input, **kwargs):
        negative_slope = 0.1
        if 'negative_slope' in kwargs:
            negative_slope = kwargs['negative_slope']
            del kwargs['negative_slope']

        if not isinstance(input,tuple):
            conv = Layers.Deconvolution(net, input, **kwargs)
            Layers.ReLU(net, conv, nout=1, in_place=True, negative_slope=negative_slope)
            return conv
        else:
            convs = Layers.Deconvolution(net, input, **kwargs)
            for conv in convs:
                Layers.ReLU(net, conv, nout=1, in_place=True, negative_slope=negative_slope)
            return convs
        return
