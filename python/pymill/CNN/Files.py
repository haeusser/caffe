#!/usr/bin/python

import os
import re
from pymill import Toolbox as tb

class IterFile:
    _filename = ''
    _iteration = ''

    def __init__(self,filename):
        self._filename = os.path.abspath(filename)
        match = re.compile('.*?_iter_([0-9]+)').match(filename)
        if not match:
            match = re.compile('.*?([0-9]{8})').match(filename)
            if not match: raise Exception('cannot extract iteration from %s' % filename)

        self._iteration = int(match.group(1))

    def filename(self):
        return self._filename

    def iteration(self):
        return self._iteration

    def delete(self,verbose=False):
        if verbose:
            tb.notice('removing %s' % self._filename, 'del')
        os.remove(self._filename)

    def __str__(self):
        return self._filename

def iterFiles(type='.solverstate', dir='.', omit=None):
    result = []

    if not os.path.isdir(dir):
        return result

    files = os.listdir(dir)
    for file in files:
        if file.endswith(type):
            if omit is not None and omit in file:
                continue
            result.append(IterFile('%s/%s' % (dir, file)))

    return sorted(result, key=lambda iter: iter.iteration())

def newestIterFile(type='.solverstate', dir='.'):
    files = iterFiles(type, dir)
    if not len(files):
        return None

    return files[-1]

def logFiles(dir):
    result = []

    if not os.path.isdir(dir):
        return result

    files = os.listdir(dir)

    for file in files:
        if file.endswith('.log'):
            result.append(IterFile('%s/%s' % (dir, file)))

    return sorted(result, key=lambda iter: iter.iteration())
