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
        args.append('flow')
        args.append('compute')
        args.append('+ldof[%s]' % self.spec().inputs()[0])
        args.append(self.img1Path()+','+self.img2Path())
        job.addCommand(self.path(),' '.join(args))
