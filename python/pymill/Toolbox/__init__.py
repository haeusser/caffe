#!/usr/bin/python

import numpy as np
import sys
import os
import re
import uuid
from scipy import misc
from fnmatch import fnmatch
import subprocess
import argparse
import argcomplete
import caffe

from Notice import notice
from Notice import noticeVerbose
from Notice import verbose

def isList(x):
    return isinstance(x, (list, tuple)) and not isinstance(x, basestring)

def makeList(x):
    if not isList(x):
        return (x,)
    return x

def get_net_output(net, name):
    return net.blobs[name].data[...].transpose(0, 2, 3, 1).squeeze()

caffe.Net.get_output = get_net_output

def caffeNet(modelFile=None, prototmp=None, inputs={}, phase=None, logging=False):
    if phase is None: phase=caffe.TEST
    if prototmp is not None:
        modelFile = tempFilename('.prototmp')
        open(modelFile,'w').write(prototmp)

    if not logging: caffe.set_logging_disabled()
    caffe.set_mode_gpu()
    net = caffe.Net(
        modelFile,
        phase
    )

    if prototmp is not None:
        os.remove(modelFile)

    for name, value in inputs.iteritems():
        value = value[np.newaxis, :, :, :].transpose(0, 3, 1, 2)
        net.blobs[name].reshape(*value.shape)
        net.blobs[name].data[...] = value

    return net

def pprint(x):
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    return pp.pprint(x)

def flowWarp(img, flow):
    import caffe
    width = img.shape[1]
    height = img.shape[0]

    print 'processing (%dx%d)' % (width, height)

    defFile = tempFilename('.prototxt')
    preprocessFile('/home/ilge/hackathon2/common/prototmp/apply_flow.prototmp', defFile, {'WIDTH': width, 'HEIGHT': height})

    caffe.set_logging_disabled()
    caffe.set_mode_gpu()
    net = caffe.Net(
        defFile,
        caffe.TEST
    )

    os.remove(defFile)

    print 'network forward pass'

    img_input = img[np.newaxis, :, :, :].transpose(0, 3, 1, 2)
    flow_input = flow[np.newaxis, :, :, :].transpose(0, 3, 1, 2)

    net.blobs['image'].reshape(*img_input.shape)
    net.blobs['image'].data[...] = img_input
    net.blobs['flow'].reshape(*flow_input.shape)
    net.blobs['flow'].data[...] = flow_input

    net.forward()
    output = net.blobs['output'].data[...].transpose(0, 2, 3, 1).squeeze()

    return output


class Parser(argparse.ArgumentParser):
    def error(self, message):
        notice('Error: %s' % message, 'failed')
        sys.exit(2)

def readableDir(prospective_dir):
    if not os.path.isdir(prospective_dir):
        raise Exception("readable_dir:{0} is not a valid path".format(prospective_dir))
    if os.access(prospective_dir, os.R_OK):
        return prospective_dir
    else:
        raise Exception("readable_dir:{0} is not a readable dir".format(prospective_dir))

def ensureDir(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def naturalKeys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]

class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('\033[91m\033[1m' + 'Error: %s\n' % message + '\033[0m')
        self.print_help()
        sys.exit(2)

def wildcardMatch(haystack, needle):
    if isList(haystack):
        return [ent for ent in haystack if fnmatch(ent, needle)]
    else:
        return [fnmatch(haystack, needle)]

def unique(input):
    output = []
    for x in input:
        if x not in output:
            output.append(x)
    return output

def system(command):
    if verbose: notice('running "%s"' % command, 'run')
    return os.system(command)

def run(command, input=''):
    p = subprocess.Popen(
        ['/bin/bash', '-c', command],
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    out, err = p.communicate(input)

    if err:
        print err

    return out

def queryYesNo(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

def largestNumber(files):
    result = []
    for f in files:
        parts = f.split('/')
        dir = ''
        for p in parts:
            if p == '.' or p =='..':
                dir += p + '/'
                continue

            bestN = -1
            bestEnt = None
            for e in os.listdir('.' if dir == '' else dir):
                match = re.compile(p).match(e)
                if match:
                    if len(match.groups()) > 0:
                        n = int(match.group(1))
                        if n > bestN:
                            bestN = n
                            bestEnt = e
                    else:
                        bestEnt = e

            if bestEnt == None:
                raise Exception('no entry found for %s in %s' % (p, f))

            dir += bestEnt + '/'

        result.append(dir)
    return result

def downsampleImage(image, f):
    import cv2
    return cv2.resize(image, (image.shape[1] / f, image.shape[0] / f), interpolation=cv2.INTER_CUBIC)

class PlotStyle:
    def __init__(self, definition=''):
        if definition.strip() == '':
            self._color=[0.0, 0.0, 0.0]
            self._lineStyle = '-'
            self._lineWidth = 1.0
            return

        match=re.compile('(([0-9]|\.)+)').match(definition)
        coeff=1.0
        prefix=None
        if match:
            prefix=match.group(1)
            coeff=float(prefix)

        name=definition
        if prefix:
            name=definition.replace(prefix,'')

        color=np.array([0.0,0.0,0.0])
        if name[0]=='y': color=np.array([1.0,1.0,0.0])
        if name[0]=='m': color=np.array([1.0,0.0,1.0])
        if name[0]=='c': color=np.array([0.0,1.0,1.0])
        if name[0]=='r': color=np.array([1.0,0.0,0.0])
        if name[0]=='g': color=np.array([0.0,1.0,0.0])
        if name[0]=='b': color=np.array([0.0,0.0,1.0])
        if name[0]=='w': color=np.array([1.0,1.0,1.0])

        color*=coeff

        linestyle = '-'
        if len(name)>1:
            linestyle=name[1:]

        self._color = color
        self._lineStyle = linestyle
        self._lineWidth = 1.0

    def lineStyle(self): return self._lineStyle
    def lineWidth(self): return self._lineWidth
    def color(self): return self._color

    def setColor(self, color):
        if color is str:
            print 'str'
            match = re.match('#([0-9A-Za-z]{2}){3}')
            self._color = (float(match(1))/255.0, float(match(2))/255.0, float(match(3))/255.0)
            return self

        self._color = color
        return self

def styleList():
    return [
        PlotStyle().setColor('#0000AA'),
        PlotStyle().setColor('#00AA00'),
        PlotStyle().setColor('#AA0000'),
        PlotStyle().setColor('#AA5500'),
        PlotStyle().setColor('#AAAAAA'),
        PlotStyle().setColor('#555555'),
        PlotStyle().setColor('#AA00AA'),
        PlotStyle().setColor('#5555FF'),
        PlotStyle().setColor('#55FF55'),
        PlotStyle().setColor('#55FFFF'),
        PlotStyle().setColor('#FF5555'),
        PlotStyle().setColor('#FF55FF'),
        PlotStyle().setColor('#FFFF55'),
        PlotStyle().setColor('#000000'),
    ]

from pymill import Config

def evaluateExpressionsInFile(inFile, outFile):
    lines = open(inFile).readlines()
    f = open(outFile,'w')

    for l in lines:
        if '[' in l:
            matches = re.compile('(\\[.*?\\])').findall(l)
            for m in matches:
                token = m[1:-1]
                value = str(eval(token))
                l = l.replace(m, value)

        f.write(l)
    f.close()

def preprocessFile(inFile, outFile, defs={}):
    settings = ''

    for key in defs:
        settings += ' -D%s=%s' % (key, defs[key])

    incPath = '-I%s' % Config.prototmpInclude

    if not system('cpp %s %s %s %s' % (inFile, outFile+'.pp', settings, incPath)) == 0:
        raise Exception('preprocessing of %s failed' % inFile)
    evaluateExpressionsInFile(outFile+'.pp', outFile)

# -------------------------------------------------------------
from IO import readPFM
from IO import writePFM
from IO import readFlow
from IO import writeFlow
from IO import readFloat
from IO import readImage
from IO import writeFloat
from IO import readDisparity
from IO import writeDisparity
from IO import readList
from IO import readTupleList
from IO import openLMDB
from IO import avprobe
from IO import avinfo
from IO import tempFilename
from IO import recv_size

from Mean import Mean
from Mean import dictionaryListMean

from PSNR import psnr
from PSNR import psnrY
from PSNR import PSNR
from PSNR import PSNRList
from PSNR import computeBasePSNRs

from Cluster import Job
from Cluster import Queue
# -------------------------------------------------------------
