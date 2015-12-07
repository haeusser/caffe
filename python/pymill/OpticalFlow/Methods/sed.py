#!/usr/bin/python
# coding: utf-8

from pymill import OpticalFlow
from pymill import Toolbox as tb
from string import Template

class Implementation(OpticalFlow.Method):
    def __init__(self):
        OpticalFlow.Method.__init__(self)
        self._outFiles.append('boundary.bin')
        self._outFiles.append('soft.float2')
        self._outFiles.append('soft.pgm')
        self._outFiles.append('thin.float2')
        self._outFiles.append('thin.pgm')

    def type(self):
        return 'boundaries'

    def computeJobImplementation(self,job):
        command = "/misc/software-lin/matlabR2013a/bin/matlab -nodesktop -nojvm -r \"sfDetect('%s'); exit\"" % self.ent().imgPath()

        job.addCommand(self.path(), command)

