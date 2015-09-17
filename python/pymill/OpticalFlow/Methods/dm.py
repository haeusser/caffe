#!/usr/bin/python

from pymill import OpticalFlow
from pymill import Toolbox as tb

class Implementation(OpticalFlow.Method):
    def __init__(self):
        OpticalFlow.Method.__init__(self)

        self._binary = True
        self._outFiles.append('confidence.float2')
        self._outFiles.append('confidence.bulk.float2')
        self._outFiles.append('flow.bulk-match')
        self._outFiles.append('flow.match')
        self._outFiles.append('flow.flo')
        self._outFilesGT.append('flow.flo.epe')
        self._outFilesGT.append('flow.flo.stat')

    def type(self):
        return 'matching'

    def computeJobImplementation(self,job):
        job.setMem(8096)

        args = []
        args.append('deepmatching')
        args.append(self.img1Path())
        args.append(self.img2Path())
        args.append('-out')
        args.append('flow.bulk-match')
        for param, value in self._spec.params().iteritems():
            if param == 's':
                if value == 'im':
                    args.append('-improved_settings')
                elif value == 'old':
                    args.append('-iccv_settings')
            elif param == 'o':
                args.append('-occlusion_check')
                args.append(value)
        job.addCommand(self.path(),' '.join(args))

        args = []
        args.append('match2flo')
        args.append('flow.bulk-match')
        args.append(self.img1Path())
        args.append('flow.flo')
        args.append('confidence.bulk.float2')
        job.addCommand(self.path(), ' '.join(args))

        args = []
        args.append('rescore')
        args.append(self.img1Path())
        args.append(self.img2Path())
        args.append('flow.bulk-match')
        args.append('flow.match')
        job.addCommand(self.path(), ' '.join(args))

        args = []
        args.append('match2flo')
        args.append('flow.match')
        args.append(self.img1Path())
        args.append('flow.flo')
        args.append('confidence.float2')
        job.addCommand(self.path(), ' '.join(args))


