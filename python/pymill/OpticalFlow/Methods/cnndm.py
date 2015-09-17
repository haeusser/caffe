#!/usr/bin/python

from pymill import OpticalFlow
from pymill import Toolbox as tb

class Implementation(OpticalFlow.Method):
    def __init__(self):
        OpticalFlow.Method.__init__(self)

        self._binary = True
        self._outFiles.append('confidence.float2')
        self._outFiles.append('confidence.no-occ.float2')
        self._outFiles.append('confidence.bulk.float2')
        self._outFiles.append('confidence.no-occ.bulk.float2')
        self._outFiles.append('feat1.float3')
        self._outFiles.append('feat2.float3')
        self._outFiles.append('flow.bulk-match')
        self._outFiles.append('flow.match')
        self._outFiles.append('flow.flo')
        self._outFiles.append('flow.no-occ.bulk-match')
        self._outFiles.append('flow.no-occ.match')
        self._outFiles.append('flow.no-occ.flo')
        self._outFiles.append('grid-params.txt')
        self._outFilesGT.append('flow.flo.epe')
        self._outFilesGT.append('flow.flo.stat')

    def type(self):
        return 'matching'

    def computeJobImplementation(self,job):
        job.setGpu(True)
        job.setMem(8096)

        args = []
        args.append('CaffeDeepMatching')
        args.append(self.img1Path())
        args.append(self.img2Path())
        args.append(self._ent.dsName())
        args.append(str(self.spec().inputs()[0]))
        for param, value in self._spec.params().iteritems():
            if param == 'lambda':
                args.append('--lambda=%s' % value)
        job.addCommand(self.path(),' '.join(args))

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

        args = []
        args.append('rescore')
        args.append(self.img1Path())
        args.append(self.img2Path())
        args.append('flow.no-occ.bulk-match')
        args.append('flow.no-occ.match')
        job.addCommand(self.path(), ' '.join(args))

        args = []
        args.append('match2flo')
        args.append('flow.no-occ.match')
        args.append(self.img1Path())
        args.append('flow.no-occ.flo')
        args.append('confidence.no-occ.float2')
        job.addCommand(self.path(), ' '.join(args))


