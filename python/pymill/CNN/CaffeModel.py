#!/usr/bin/python

import caffe
import os
import sys
import argparse
from pymill import Toolbox as tb
import numpy as np

def readParams(filename):
    f = open(filename,'rb')
    param = caffe.proto.caffe_pb2.NetParameter()
    param.ParseFromString(f.read())
    f.close()
    return param

def writeParams(param, filename):
    f = open(filename, 'wb')
    f.write(param.SerializeToString())
    f.close()

def fixBlob(blob):
    if len(blob.shape.dim) == 4:
        blob.num = blob.shape.dim[0]
        blob.channels = blob.shape.dim[1]
        blob.height = blob.shape.dim[2]
        blob.width = blob.shape.dim[3]
    else:
        blob.num = blob.shape.dim[0]
        blob.channels = 1
        blob.height = 1
        blob.width = 1
    return blob

class Model:
    def __init__(self, filename=None):
        self._param = caffe.proto.caffe_pb2.NetParameter()

        if filename is not None:
            self.readFrom(filename)

    def param(self): return self._param

    def readFrom(self, filename):
        self._param = readParams(filename)

    def writeTo(self, filename):
        writeParams(self._param, filename)

    def getLayer(self, layerName):
        for l in self._param.layer:
            if l.name == layerName:
                return l

        raise Exception('Layer %s not found' % layerName)

    def getLayerBlobs(self, layerName):
        return [caffe.io.blobproto_to_array(fixBlob(blob)) for blob in self.getLayer(layerName).blobs]

    def setLayerBlobs(self, layer, blobs):
        if isinstance(layer, str):
            layer = self.getLayer(layer)

        i = 0
        for blob in blobs:
            for j in range(0, len(layer.blobs[i].data)):
                del layer.blobs[i].data[0]
            layer.blobs[i].data.extend(blob.astype(float).flat)
            for j in range(0,len(layer.blobs[i].shape.dim)):
                layer.blobs[i].shape.dim[j] = blob.shape[j]
            fixBlob(layer.blobs[i])
            i += 1


if __name__ == "main":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', prog='caffemodel')

    # list-blobs
    subparser = subparsers.add_parser('list-blobs', help='list blobs')
    subparser.add_argument('caffemodel', help='.caffemodel file')

    # list-layers
    subparser = subparsers.add_parser('list-layers', help='list layers')
    subparser.add_argument('caffemodel', help='.caffemodel file')

    # add-prefix
    subparser = subparsers.add_parser('add-prefix', help='prefix blobs')
    subparser.add_argument('caffemodel', help='.caffemodel file')
    subparser.add_argument('prefix', help='prefix to use')

    args = parser.parse_args()
    param = readParams(args.caffemodel)

    if args.command == 'list-layers':
        for l in param.layer:
            print l.name

    elif args.command == "list-blobs":
        for l in param.layer:
            n = len(l.blobs)
            if n>0:
                print '%s:' % l.name
                i = 0
                for blob in l.blobs:
                    print "    * blob %s: " % i,
                    for dim in blob.shape.dim:
                        print "%d" % dim,
                    print
                    i += 1

    elif args.command == "add-prefix":
        for l in param.layer:
            n = len(l.blobs)
            if n>0:
                l.name = args.prefix + l.name
                print '%s:' % l.name
                i = 0
                for blob in l.blobs:
                    print "    * blob %s: " % i,
                    for dim in blob.shape.dim:
                        print "%d" % dim,
                    print
                    i += 1

        writeParams(param, args.caffemodel)



