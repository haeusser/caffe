#!/usr/bin/python

from pymill import OpticalFlow
from pymill import Toolbox as tb

class Implementation(OpticalFlow.Method):
    def __init__(self):
        OpticalFlow.Method.__init__(self)

        self._binary = True
        self._outFiles.append('confidence.float2')
        self._outFiles.append('flow.flo')
        self._outFilesGT.append('flow.flo.epe')
        self._outFilesGT.append('flow.flo.stat')

    def type(self):
        return 'matching'

    def computeJobImplementation(self, job):
        job.setMem(8096)

        args = []
        args.append('flow')
        args.append('compute')
        method="+hog_nn"
        for param, value in self._spec.params().iteritems():
            if param == 'r':
                method += "@r=" + value
        args.append(method)
        args.append(self.img1Path()+','+self.img2Path())
        job.addCommand(self.path(),' '.join(args))
