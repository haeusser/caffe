#!/usr/bin/python
from _codecs import raw_unicode_escape_decode

import argparse
import numpy as np
from Environment import Environment
from pymill import Toolbox as tb
from scipy import misc
import cv2
import Environment

class Test:
    def __init__(self, name, iterations):
        self._env = Environment.Environment()
        self._env.init()
        self._iter = self._env.stateFiles()[-1].iteration()
        self._output = False
        self._variables = {}
        self._name = name
        self._iterations = iterations

    def setOutput(self, value): self._output = value
    def setIterations(self, value): self._iterations = value
    def setIter(self, value): self._iter = value

    def parseArgs(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--iter', help='iteration', default=-1, type=int)
        parser.add_argument('--output', help='output', action='store_true')
        parser.add_argument('--verbose', help='verbose', action='store_true')

        args = parser.parse_args()
        tb.verbose = args.verbose

        if args.iter != -1: self._iter = args.iter
        self._output = args.output

    def set(self, key, value):
        self._variables[key] = value

    def runProto(self, proto):
        defFile = proto
        modelFile, iter = self._env.getModelFile(self._iter)

        print 'testing for iteration %d ...' % self._iter

        if self._output:
            dir = 'output_%s_%d' % (self._name, self._iter)
            tb.system('mkdir -p %s' % dir)
            self._variables['TEST_OUTPUT'] = 1
            self._variables['TEST_OUTPUT_DIR'] = '"\\"%s\\""' % dir

        self._env.makeScratchDir()
        defPrototxt = self._env.prototxt(defFile, 'scratch', self._variables)
        print defFile, defPrototxt
        tb.system('%s test -weights %s -model %s -gpu 0 -iterations %d 2>&1' % (Environment.caffeBin(), modelFile, defPrototxt, self._iterations))




