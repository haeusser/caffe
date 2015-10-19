#!/usr/bin/python

import caffe
import os
import sys
import argparse
from pymill import Toolbox as tb

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

f = open(args.caffemodel,'rb')
param = caffe.proto.caffe_pb2.NetParameter()
param.ParseFromString(f.read())
f.close()

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

    f = open(args.caffemodel, 'wb')
    f.write(param.SerializeToString())
    f.close()



