#!/usr/bin/python
# coding: utf-8

import os
import re

class UnaryEntity:
    format = '%04d'

    def __init__(self, id, dsName, sceneName, name, imgPath, label=1):
        self._id = int(id)
        self._dsName = dsName
        self._sceneName = sceneName
        self._name = name
        self._label = label
        self._imgPath = imgPath

    def binary(self):
        return False

    def unary(self):
        return True

    def inFile(self, spec, file):
        path = '%s/%s/%s' % (self.path(), str(spec), file)
        if not os.path.isfile(path):
            raise Exception('input file not found: %s' % path)
        return path

    def outFile(self, spec, file):
        path = '%s/%s/%s' % (self.path(), str(spec), file)
        dir = '/'.join(path.split('/')[:-1])
        if not os.path.isdir(path):
            os.system('mkdir -p %s' % dir)
        return path

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

    def path(self):
        return self.imgPath().replace('.ppm','')

    def chdir(self):
        os.chdir(self.path())

    def imgPath(self):
        return self._imgPath

    def imgFilename(self):
        return self._imgPath.split('/')[-1]

    def parentPath(self):
        return '/'.join(self._imgPath.split('/')[:-1])

    def chdirParent(self):
        os.chdir(self.parentPath())

    def formattedFrameNumber(self):
        parts = self._imgPath.split('/')
        filename = parts[-1]
        return self.format % int(re.compile("([0-9]+)").match(filename).group(1))

    def figureLabel(self):
        return '%s/%s' % self._sceneName + self.formattedFrameNumber()

    def bind(self,spec):
        from Methods import methods
        from Method import Method
        if not spec.name() in methods:
            method = Method()
            method.setEnt(self)
            method.setSpec(spec)
            return method

        exec('method = Methods.%s.Implementation()' % spec.name())

        method.setEnt(self)
        method.setSpec(spec)

        if not method.unary():
            raise Exception('tried to bind non-unary method <%s> to unary entity <%s>' % (spec.name(), self.name()))

        return method

    def __str__(self):
        parts = []

        parts.append(self.formattedId())
        parts.append(self._dsName)
        parts.append(self._sceneName)
        parts.append(self._name)
        parts.append(str(self._label))
        parts.append(self._imgPath)

        return ' '.join(parts)
