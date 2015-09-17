#!/usr/bin/python
# coding: utf-8

import os
from UnaryEntity import UnaryEntity
from BinaryEntity import BinaryEntity

class Dataset:
    def __init__(self, name):
        self._name = name
        self._bEnts = []
        self._uEnts = []

        for entry in open('/home/ilge/data/%s.bents'%name).readlines():
            self._bEnts.append(BinaryEntity(entry))

        files = []
        id = 0
        for ent in self._bEnts:
            if not ent.img1Path() in files:
                self._uEnts.append(UnaryEntity(id, ent.dsName(), ent.sceneName(), ent.name(), ent.img1Path(), ent.label()))
                files.append(ent.img1Path())
                id += 1
            if not ent.img2Path() in files:
                self._uEnts.append(UnaryEntity(id, ent.dsName(), ent.sceneName(), ent.name(), ent.img2Path(), ent.label()))
                files.append(ent.img2Path())
                id += 1

    def uents(self):
        return self._uEnts

    def bents(self):
        return self._bEnts

    def name(self):
        return self._name

    @staticmethod
    def names():
        list=[]
        for f in os.listdir('/home/ilge/data'):
            if f.endswith('.bents'):
                list.append(f.replace('.bents',''))

        return sorted(list)

    @staticmethod
    def infer():
        dsNames = Dataset.names()

        path = os.getcwd().split('/')
        for part in path:
            part = part.split('_')[0]
            for ds in dsNames:
                if part == ds:
                    return Dataset(ds)
        return None

    def __repr__(self):
        return self._name

    def figureLabels(self):
        return [e.figureLabel() for e in self._bEnts]

    def restrictTo(self, label):
        if label == -1: return
        self._uEnts = [uEnt for uEnt in self._uEnts if uEnt.label() == label]
        self._bEnts = [bEnt for bEnt in self._bEnts if bEnt.label() == label]

    def limitTo(self, limit):
        if limit == '': return

        if '-' in limit:
            parts = limit.split('-')
            start = parts[0]
            end = parts[1]

            if start < 0: start = 0
            if end > len(self._uEnts): end = len(self._uEnts)

            self._uEnts = self._uEnts[start:end]
            self._bEnts = self._bEnts[start:end]

        else:
            limit=int(limit)
            if len(self._uEnts) > limit: self._uEnts = self._uEnts[0:limit]
            if len(self._bEnts) > limit: self._bEnts = self._bEnts[0:limit]
















    # def integrate(self,name,folder):
    #     for ent in self._bEnts:
    #         ent.integrate(name,'%s/%s'%(folder,ent.sceneName()))

    # def figureLabels(self):
    #     l = []
    #
    #     for e in self._bEnts:
    #         l.append(e.figureLabel())
    #
    #     return l
    #
    # def epeList(self,method):
    #     list=[]
    #
    #     for ent in self._bEnts:
    #         list.append(ent.epe(method))
    #
    #     return list
