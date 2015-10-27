#!/usr/bin/python

import os
from pymill import OpticalFlow
from pymill import Toolbox as tb

class Implementation(OpticalFlow.Method):
    def __init__(self):
        OpticalFlow.Method.__init__(self)

        self._binary = True
        self._outFiles.append('disp.float3')
        self._outFilesGT.append('disp.float3.epe')

    def type(self):
        return 'disp'

    def computeJobImplementation(self,job):
        args = []
        args.append('/misc/lmbraid17/sceneflownet/haeusserp/hackathon-code/lecun.py')
        args.append(self.img1Path())
        args.append(self.img2Path())
        args.append(os.path.join(self.path(), 'disp.float3'))

        job.addCommand(self.path(),' '.join(args))
