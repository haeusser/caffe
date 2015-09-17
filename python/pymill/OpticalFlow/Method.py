#!/usr/bin/python
# coding: utf-8
from distutils.command.config import config

import os
import shutil
from termcolor import colored
from pymill import Toolbox as tb


class Method:
    def __init__(self):
        self._ent = None
        self._spec = None
        self._outFiles = ['info']
        self._outFilesGT = []
        self._type = None
        self._binary = False
        pass

    def unary(self): return not self._binary
    def binary(self): return self._binary

    def setEnt(self,ent): self._ent = ent
    def ent(self): return self._ent

    def setSpec(self,spec): self._spec=spec
    def spec(self): return self._spec

    def path(self):
        if self._ent.binary():
            return '%s/%s' % (self._ent.path(self._spec.direction()), str(self._spec))
        else:
            return '%s/%s' % (self._ent.path(), str(self._spec))

    def file(self,name):
        return self._ent.outFile(self._spec, name)

    def imgPath(self):
        return self._ent.imgPath()

    def img1Path(self):
        if self._ent.binary():
            return self._ent.img1Path(self._spec.direction())
        else:
            return self._ent.imgPath()

    def img2Path(self):
        return self._ent.img2Path(self._spec.direction())

    def flowGTPath(self):
        return self._ent.flowGTPath(self._spec.direction())

    def occGTPath(self, dir='+'):
        return self._ent.occGTPath(self._spec.direction())

    def mbdistGTPath(self, dir='+'):
        return self._ent.mbdistGTPath(self._spec.direction())

    def checkOut(self,verbose=True):
        if not os.path.exists(self.path()):
            if verbose: print colored('missing %s' % self.path(),'red')
            return False

        for file in self._outFiles:
            path = '%s/%s' % (self.path(),file)

            if not os.path.isfile(path):
                if verbose: print colored('missing %s' % path,'red',attrs={'bold':1})
                return False

        return True

    def clean(self):
        if os.path.isdir(self.path()):
            tb.noticeVerbose('removing %s' % self.path(), 'remove')
            shutil.rmtree(self.path())
            return True
        return False

    def computeJobImplementation(self,job):
        raise Exception('computeJobImplementation not implemented for %s' % self._spec)

    def updateJobImplementation(self,job):
        self.computeJobImplementation(job)

    def makeComputeJob(self,job):
        job.setLog(self.file('log.txt'))
        tb.noticeVerbose('creating compute job for <%s>' % self.img1Path(),'run')

        job.addCommand(self.path(), "echo type=%s > info" % self.type())
        self.computeJobImplementation(job)
        self.epeJobImplementation(job)

    def makeUpdateJob(self,job):
        if not self.checkOut(False):
            job.setLog(self.file('log.txt'))
            job.addCommand(self.path(), "echo type=%s > info" % self.type())
            self.updateJobImplementation(job)

    def makeEpeJob(self,job):
        job.setLog(self.file('log.txt'))
        self.epeJobImplementation(job)

    def epeJobImplementation(self,job):
        if self.type() == 'matching' or self.type() == 'flow':
            cmd = 'echo "%s" | FlowStat --make-stat --make-epe' % self.flowStatParams()
            job.addCommand(self.path(), cmd)

    def type(self):
        infoPath = '%s/info' % self.path()
        if os.path.isfile(infoPath):
            lines = open(infoPath,'r').readlines()
            for line in lines:
                if line.startswith("type"):
                    parts=line.split('=')
                    return parts[2].strip()

    def flowStatParams(self):
        confidence = None
        if self.type() == 'matching':
            confidence = self.file('confidence.float2')

        return '%s %s %s %s %s %s %s' % (
            self.img1Path(),
            self.img2Path(),
            self.file('flow.flo'),
            confidence,
            self.flowGTPath(),
            self.occGTPath(),
            self.mbdistGTPath()
        )


