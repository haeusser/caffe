#!/usr/bin/python
# coding: utf-8

import os
import re

class BinaryEntity:
    format = '%04d'

    def __init__(self,str):
        parts = str.split()

        self._id = int(parts[0].strip())
        self._dsName = parts[1].strip()
        self._sceneName = parts[2].strip()
        self._name = parts[3].strip()
        self._label = int(parts[4].strip())
        self._img1Path = parts[5].strip()
        self._img2Path = parts[6].strip()

        self._fwdFlowGTPath = None
        self._fwdOccGTPath = None
        self._fwdMbdistGTPath = None

        self._bwdFlowGTPath = None
        self._bwdOccGTPath = None
        self._bwdMbdistGTPath = None

        if len(parts) > 7:   self._fwdFlowGTPath = parts[7].strip()
        if len(parts) > 8:   self._fwdOccGTPath = parts[8].strip()
        if len(parts) > 9:   self._fwdMbdistGTPath = parts[9].strip()
        if len(parts) > 10:  self._bwdFlowGTPath = parts[10].strip()
        if len(parts) > 11:  self._bwdOccGTPath = parts[11].strip()
        if len(parts) > 12:  self._bwdMbdistGTPath = parts[12].strip()

    def binary(self):
        return True

    def unary(self):
        return False

    def inFile(self, spec, file):
        paths = []
        if spec.direction() == '+':    paths.append('%s/%s/%s' % (self.path1(), str(spec), file))
        elif spec.direction() == '-':  paths.append('%s/%s/%s' % (self.path2(), str(spec), file))
        else:
            paths.append('%s/%s/%s' % (self.path1(), str(spec), file))
            paths.append('%s/%s/%s' % (self.path2(), str(spec), file))

        #for path in paths:
        #    if not os.path.isfile(path):
        #        raise Exception('input file not found: %s' % path)

        if len(paths) == 1:
            return paths[0]

        return paths

    def outFile(self,spec,file):
        paths = []
        if spec.direction() == '+':    paths.append('%s/%s/%s' % (self.path1(), str(spec), file))
        elif spec.direction() == '-':  paths.append('%s/%s/%s' % (self.path2(), str(spec), file))
        else:
            paths.append('%s/%s/%s' % (self.path1(), str(spec), file))
            paths.append('%s/%s/%s' % (self.path2(), str(spec), file))

        for path in paths:
            dir = '/'.join(path.split('/')[:-1])
            if not os.path.isdir(path):
                os.system('mkdir -p %s' % dir)

        if len(paths) == 1:
            return paths[0]

        return paths

    def id(self):
        return self._id

    def formattedId(self):
        return self.format % self._id

    def dsName(self):
        return self._dsName

    def sceneName(self):
        return self._sceneName

    def name(self):
        return self._name

    def label(self):
        return self._label

    def detailedName(self):
        return '%s/%s/%s' % (self._dsName, self._sceneName, self._name.replace('.ppm',''))

    def path1(self):
        return self.img1Path().replace('.ppm','')

    def path2(self):
        return self.img2Path().replace('.ppm','')

    def path(self, dir='+'):
        if dir == '-':  return self.path2()
        else:           return self.path1()

    def chdir(self):
        os.chdir(self.path())

    def img1Path(self, dir='+'):
        if dir == '-':  return self._img2Path
        else:           return self._img1Path

    def img2Path(self, dir='+'):
        if dir == '-':  return self._img1Path
        else:           return self._img2Path

    def flowGTPath(self, dir='+'):
        if dir == '-':  return self._bwdFlowGTPath
        else:           return self._fwdFlowGTPath

    def occGTPath(self, dir='+'):
        if dir == '-':  return self._bwdOccGTPath
        else:           return self._fwdOccGTPath

    def mbdistGTPath(self, dir='+'):
        if dir == '-':  return self._bwdMbdistGTPath
        else:           return self._fwdMbdistGTPath

    def img1Filename(self, dir='+'):
        if dir == '-':  return self._img2Path.split('/')[-1]
        else:           return self._img1Path.split('/')[-1]

    def img2Filename(self, dir='+'):
        if dir == '-':  return self._img1Path.split('/')[-1]
        else:           return self._img2Path.split('/')[-1]

    def parentPath(self):
        return '/'.join(self._img1Path.split('/')[:-1])

    def chdirParent(self):
        os.chdir(self.parentPath())

    def idFlo(self,ext):
        if len(ext):
            if ext.startswith('.'): ext = ext[1:]
            if ext.endswith('.'):   ext = ext[:-1]
            return self.formattedId() + '.' + ext + '.flo'

        return self.formattedId() + '.flo'

    def idLowresFlo(self,ext=''):
        if len(ext):
            if ext.startswith('.'): ext = ext[1:]
            if ext.endswith('.'):   ext = ext[:-1]
            return self.formattedId() + '.' + ext + '.lowres.flo'

        return self.formattedId() + '.lowres.flo'

    def formattedFrameNumber(self):
        parts = self._img1Path.split('/')
        filename = parts[-1]
        return self.format % int(re.compile("[^0-9]*([0-9]+).*").match(filename).group(1))

    def figureLabel(self):
        return '%s/%s' % (self._sceneName, self.formattedFrameNumber())

    def bind(self, spec):
        from pymill.OpticalFlow import Methods
        from pymill.OpticalFlow.Method import Method

        if not spec.name() in Methods.methods:
            method = Method()
            method.setEnt(self)
            method.setSpec(spec)
            return method

        exec('method = Methods.%s.Implementation()' % spec.name())

        method.setEnt(self)
        method.setSpec(spec)

        if not method.binary():
            raise Exception('tried to bind non-binary method <%s> to binary entity <%s>' % (spec.name(), self.name()))

        return method

    def __str__(self):
        parts = []

        parts.append(self.formattedId())
        parts.append(self._dsName)
        parts.append(self._sceneName)
        parts.append(self._name)
        parts.append(str(self._label))
        parts.append(self._img1Path)
        parts.append(self._img2Path)

        parts.append(self._fwdFlowGTPath if self._fwdFlowGTPath else None)
        parts.append(self._fwdOccGTPath if self._fwdOccGTPath else None)
        parts.append(self._fwdMbdistGTPath if self._fwdMbdistGTPath else None)
        parts.append(self._bwdFlowGTPath if self._bwdFlowGTPath else None)
        parts.append(self._bwdOccGTPath if self._bwdOccGTPath else None)
        parts.append(self._bwdMbdistGTPath if self._bwdMbdistGTPath else None)

        while not parts[-1]: parts = parts[:-1]

        return ' '.join(parts)
