#!/usr/bin/python

from CaffeAdapter import *
from Blocks import Blocks
from Data import *
from caffe.proto import caffe_pb2 as Proto
from Util import *
import Dataset
import sys

def argVars():
    d = {}

    params = sys.argv[1:]

    for p in params:
        if not '=' in p:
            raise Exception('Parameter %s is not of the form key=value' % p)
        k, v = p.split('=')
        d[k] = v

    return d

args = None


def param(name, default=None):
    global args
    if args is None:
        args = argVars()

    if name in args: return args[name]
    return default
