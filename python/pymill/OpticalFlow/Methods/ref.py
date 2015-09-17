#!/usr/bin/python
# coding: utf-8

from pymill import OpticalFlow
from pymill import Toolbox as tb
from string import Template
import os

class Implementation(OpticalFlow.Method):
    def __init__(self):
        OpticalFlow.Method.__init__(self)
        self._outFiles.append('flow.flo')
        self._binary = True

    def type(self):
        return 'flow'

    def compute(self,queue):
        OpticalFlow.Method.compute(self,queue)

        ent = self.ent()
        input = self.spec().inputs()[0]
        inputPath = ent.inFile(input,"lowres.flo")
        if not os.path.exists(inputPath):
            raise Exception('input does not exist: %s' % inputPath)

        params = self.spec().params()
        paramStr = ''

        if 'b' in params and params['b']=='1':
            paramStr+= ' --boundaries='+ent.boundaryPath()

        if 'delta' in params:
            paramStr+= ' --delta='+params['delta']

        if 'beta' in params:
            paramStr+= ' --beta='+params['beta']

        if 'sigma' in params:
            paramStr+= ' --sigma='+params['sigma']

        os.system("rm -rf %s" % ent.outFile(str(self.spec()),'flow.flo'))

        if ent.flowGTPath!=None:
            temp = Template('VarFlowRefine $img1 $img2 $flo $out $params --groundtruth=$groundtruth --make-epe --make-stat');
            command = temp.safe_substitute(
                img1=ent.img1Path(),
                img2=ent.img2Path(),
                flo=inputPath,
                out=ent.outFile(str(self.spec()),'flow.flo'),
                groundtruth='%s:%s:%s' % (ent.flowGTPath(), ent.occGTPath(), ent.mbdistGTPath()),
                params=paramStr
            )
        else:
            temp = Template('VarFlowRefine $img1 $img2 $flo $out $params');
            command = temp.safe_substitute(
                img1=ent.img1Path(),
                img2=ent.img2Path(),
                flo=inputPath,
                out=ent.outFile(str(self.spec()),'flow.flo'),
                params=paramStr
            )

        job = tb.Job()
        job.setPath(self.path())
        job.addCommand(command)
        queue.postJob(job)

    def update(self,queue):
        pass
