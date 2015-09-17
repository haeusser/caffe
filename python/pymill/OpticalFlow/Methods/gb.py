#!/usr/bin/python
# coding: utf-8

from pymill import OpticalFlow
from pymill import Toolbox as tb
from string import Template

class Implementation(OpticalFlow.Method):
    def __init__(self):
        OpticalFlow.Method.__init__(self)
        self._outFiles.append('thin.pgm')
        self._outFiles.append('soft.pgm')
        self._outFiles.append('thin.float2')
        self._outFiles.append('data.float3')

    def type(self):
        return 'boundaries'

    def compute(self,queue):
        OpticalFlow.Method.compute(self,queue)

        temp = Template('/misc/software-lin/matlabR2013a/bin/matlab -nodesktop -nojvm -r "gbDetect(\'$imagePath\',1); exit"')

        command = temp.safe_substitute(
            imagePath=self.ent().img1Path()
        )

        job = tb.Job()
        job.setPath(self.path())
        job.addCommand(command)
        queue.postJob(job)

