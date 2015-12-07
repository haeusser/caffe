#!/usr/bin/python

from pymill import OpticalFlow
from pymill import Toolbox as tb

class Implementation(OpticalFlow.Method):
    def __init__(self):
        OpticalFlow.Method.__init__(self)

        self._binary = True
        self._outFiles.append('flow.flo')
        self._outFilesGT.append('flow.flo.epe')
        self._outFilesGT.append('flow.flo.stat')

    def type(self):
        return 'flow'

    def computeJobImplementation(self,job):
        args = []
        args.append('epicflow')
        args.append(self.img1Path())
        args.append(self.img2Path())
        args.append(self._ent.inFile(OpticalFlow.Specification('sed'), 'boundary.bin')[0 if self.spec().direction() == '+' else 1])
        args.append(self._ent.bind(self.spec().inputs()[0]).file('flow.match'))
        args.append('flow.flo')

        if self._ent.dsName().startswith('sintel'):     args.append('-sintel')
        if self._ent.dsName().startswith('kitti'):      args.append('-kitti')
        if self._ent.dsName().startswith('middlebury'): args.append('-middlebury')

        job.addCommand(self.path(), ' '.join(args))


