#!/usr/bin/python
# coding: utf-8

import os
import numpy as np
import re
import pylab as plt
from pymill import Toolbox as tb

from BinaryEntity import BinaryEntity
from Specification import Specification
from Dataset import Dataset
from Method import Method
from downsample import downsample as downsampleMedianStub

def downsampleMedian(flow, f):
    if f == 1:
        return flow

    if not flow.flags['C_CONTIGUOUS']:
        flow = flow.copy(order='C')

    return downsampleMedianStub(flow, f)

class EpeStatistic:
    class Value:
        def __init__(self,str):
            parts = str.split(':')
            self._name = parts[0].strip()

            self._min = None
            self._max = None
            self._sum = None
            self._N = None
            self._mean = None
            self._contr = None

            assignments = parts[1].split(',')
            for assignment in assignments:
                if assignment.strip() == '': continue

                parts = assignment.split('=')
                key = parts[0].strip()
                value = parts[1].strip()

                if   key == 'min': self._min = float(value)
                elif key == 'max': self._max = float(value)
                elif key == 'sum': self._sum = float(value)
                elif key == 'mean': self._mean = float(value)
                elif key == 'N': self._N = int(value)
                elif key == 'contr': self._contr = float(value.replace('%',''))

        def name(self): return self._name
        def min(self): return self._min
        def max(self): return self._max
        def sum(self): return self._sum
        def N(self): return self._N
        def contr(self): return self._contr
        def mean(self): return self._mean

    def __init__(self, file = None):
        self._values={}
        if file == None: return

        for line in open(file).readlines():
            if line.strip() == '': continue

            value = EpeStatistic.Value(line)
            self._values[value.name()] = value

    def __getitem__(self, key):
        return self._values[key]

class EpePlot:
    def __init__(self,ds):
        self._ds = ds
        self._curves = []

    def addEpeFile(self, name, file, style):
        with open(file) as f: lines = [float(line) for line in f.readlines() if line.strip() != '' and line[0:3] != 'avg']

        self._curves.append(
            {'name':name, 'values': np.array(lines), 'style': style}
        )

    def addEpeValues(self, name, data, style):
        data = [float(x) for x in data]
        self._curves.append(
            {'name':name, 'values': np.array(data), 'style': style}
        )

    def addMethods(self, methods, type='all'):
        epe = epeLists(methods, [self._ds], type)

        for m in methods:
            self.addEpeValues(m, epe[str(m)][:-1], {})

    def makeDiff(self):
        values = self._curves[0]['values']

        for c in self._curves:
            c['values'] = np.subtract(c['values'], values)

    def setColorDef(self, definition):
        if definition.strip() == '':
            return

        i=0
        for method in definition.split(':'):
            match=re.compile('(([0-9]|\.)+)').match(method)
            coeff=1.0
            prefix=None
            if match:
                prefix=match.group(1)
                coeff=float(prefix)

            name=method
            if prefix:
                name=method.replace(prefix,'')

            color=np.array([0.0,0.0,0.0])
            if name[0]=='y': color=np.array([1.0,1.0,0.0])
            if name[0]=='m': color=np.array([1.0,0.0,1.0])
            if name[0]=='c': color=np.array([0.0,1.0,1.0])
            if name[0]=='r': color=np.array([1.0,0.0,0.0])
            if name[0]=='g': color=np.array([0.0,1.0,0.0])
            if name[0]=='b': color=np.array([0.0,0.0,1.0])
            if name[0]=='w': color=np.array([1.0,1.0,1.0])

            color*=coeff

            self._curves[i]['style']['color']=color
            i+=1

    def plot(self):
        plt.figure()
        plt.title("EPEs for "+self._ds.name())

        x=np.arange(0,len(self._ds.bents()))

        plt.xticks(x, self._ds.figureLabels(), rotation='vertical')

        legend = []
        for c in self._curves:
            if 'color' in c['style']:
                plt.plot(x, c['values'], color=c['style']['color'].tolist())
            else:
                plt.plot(x, c['values'])
            legend.append(c['name'])

        plt.legend(legend)


def epeListAvg(filename):
    with open(filename) as f: lines = [line for line in f.readlines() if line[0:3] == 'avg']
    return float(lines[0].replace('avg=', '').strip())

def epeList(filename):
    return [line for line in open(filename).readlines() if line.strip() != '']

def showSummary(datasets,methods):
    for ds in datasets:
        print '--------------------------  %s -------------------------  '%ds

        for m in methods:
            name=m["name"]
            epe_file=m["epe-list"].safe_substitute(
                dataset=ds
            )

            print "%80s:   %f"%(name,epeListAvg(epe_file))

def plotSummary(datasets,methods):
    for ds in datasets:
        names = []
        values = []
        colors = []
        for m in methods:
            names.append(m["name"])
            epe_file=m["epe-list"].safe_substitute(
                dataset=ds
            )
            values.append(epeListAvg(epe_file))

        plt.figure()
        x = np.arange(len(values))

        barlist = plt.bar(x,values)
        for i in range(0,len(barlist)):
            barlist[i].set_color(methods[i]["style"]["color"])
        plt.xticks(x+0.4,names,rotation='vertical')
        plt.title(ds)
        plt.subplots_adjust(bottom=0.3)
        plt.plot()

def plotEpeLists(datasets,methods):
    for ds in datasets:
        plot = Plot(Dataset(ds))

        for m in methods:
            name = m["name"]
            epe_file=m["epe-list"].safe_substitute(
                dataset=ds
            )

            print epe_file
            plot.addEpeFile(name,epe_file,m["style"])

        plot.plot()

def datasetWildcardMatch(name1, name2):
    parts1 = name1.split('.')
    parts2 = name2.split('.')

    lastWild = False
    match = True
    for i in range(0,min(len(parts1),len(parts2))):
        lastWild = False
        if parts1[i] == '*':
            lastWild = True
        else:
            if parts1[i] != parts2[i]:
                match = False
                break

    if len(parts1) != len(parts2) and not lastWild:
        match = False

    return match

def epeLists(methods,datasets,type='all'):
    epe = {}
    for m in methods:
        entries = []
        for ds in datasets:
            for ent in ds.bents():
                entries.append(ent.bind(m).flowStatParams())

        cmd = 'FlowStat --epe-type=%s' % (
            type,
        )

        epe[str(m)] = [line for line in tb.run(cmd, '\n'.join(entries)).split('\n') if line.strip() != '']

    return epe

def expandDataset(buf, label=-1, limit=''):
    list = []

    parts = buf.split(',')
    for part in parts:
        for name in Dataset.names():
            if datasetWildcardMatch(part, name):
                ds = Dataset(name)
                ds.restrictTo(label)
                ds.limitTo(limit)
                list.append(ds)

    return list







































    #
    #
    # def integrateMatMatches(self,name,folder):
    #     for ent in self._entities:
    #         srcFile='%s/%s' % (folder, ent.sceneName())
    #         print "integrating %s/%s" % (srcFile,name)
    #         ent.integrateMatMatches(name, srcFile)
    #
    #
    #













    #
    # def integrate(self,name,srcPath):
    #     srcPath = os.path.abspath(srcPath)
    #     dstPath='%s/%s' % (self.path(),name)
    #     try:
    #         os.mkdir(dstPath)
    #     except:
    #         pass
    #     dstFlowFile='%s/flow.flo'%dstPath
    #     dstInfoFile='%s/info'%dstPath
    #
    #     srcFile='%s/%s.flo' % (srcPath,self.name())
    #
    #     os.system('rm -f %s; ln -s %s %s' % (dstFlowFile,srcFile,dstFlowFile))
    #     open(dstInfoFile,'w').write("type=flow\n")
    #
    #     print srcFile,dstFlowFile,dstInfoFile
    #
    # def integrateLocal(self,name,inext):
    #     srcPath = os.path.abspath(self.idFlo(inext))
    #
    #     dstPath='%s/%s' % (self.path(),name)
    #     try:
    #         os.mkdir(dstPath)
    #     except:
    #         pass
    #     dstFlowFile='%s/flow.flo'%dstPath
    #     dstInfoFile='%s/info'%dstPath
    #
    #     os.system('rm -f %s; ln -s %s %s' % (dstFlowFile,srcPath,dstFlowFile))
    #     open(dstInfoFile,'w').write("type=flow\n")
    #
    #     if os.path.exists(srcPath):
    #         srcPath = os.path.abspath(self.idLowresFlo())
    #         dstFlowFile='%s/lowres.flo'%dstPath
    #         print 'rm -f %s; ln -s %s %s' % (dstFlowFile,srcPath,dstFlowFile)
    #         os.system('rm -f %s; ln -s %s %s' % (dstFlowFile,srcPath,dstFlowFile))
    #
    #     print srcPath,dstFlowFile,dstInfoFile
    #
    #
    # def integrateMatMatches(self,name,srcPath):
    #     dstPath='%s/%s' % (self.path(),name)
    #
    #     try:
    #         os.mkdir(dstPath)
    #     except:
    #         pass
    #
    #     dstFlowFile='%s/flow.flo'%dstPath
    #     dstConfFile='%s/confidence.float2'%dstPath
    #     dstInfoFile='%s/info'%dstPath
    #
    #     srcFile='%s/%s.mat' % (srcPath,self.name())
    #
    #     mat=scipy.io.loadmat(srcFile)
    #
    #     flow=mat['Fwd_sparce']
    #     conf=mat['Score']
    #
    #     i,j = np.where(conf==0)
    #     flow[i,j,:]=0
    #
    #     tb.writeFlow(dstFlowFile,flow)
    #     tb.writeFloat(dstConfFile,conf)
    #
    #     open(dstInfoFile,'w').write("type=matching\n")
    #
    # def epe(self,method):
    #     path='%s/%s/epe.txt' % (self.path(),method)
    #
    #     epe=float(open(path).readline())
    #     return epe






