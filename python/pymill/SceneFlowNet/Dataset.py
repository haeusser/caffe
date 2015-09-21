#!/usr/bin/python

import os
import caffe
from scipy import misc
from pymill import Toolbox as tb
import numpy as np
from pymill import OpticalFlow
import time
from collections import OrderedDict

def readBents(bents):
    list = tb.readTupleList(bents)

    clips = []

    clip = []
    current = None
    for ent in list:
        if ent[2] != current:
            if len(clip): clips.append(clip)
            clip = []
            current = ent[2]
        clip.append(ent)

    if len(clip): clips.append(clip)


    data = OrderedDict()

    for clip in clips:
        sceneName = clip[0][2]
        data[sceneName] = []
        for frame in clip:
            data[sceneName].append({'cleanImageL': frame[5], 'finalImageL': frame[5].replace('clean','final'), 'forwardFlowL': frame[7]})
        data[sceneName].append({'cleanImageL': frame[6], 'finalImageL': frame[6].replace('clean','final')})

    return data

class Frame:
    def __init__(self,
                 first=False,
                 last=False,
                 cleanImageL=None,
                 cleanImageR=None,
                 finalImageL=None,
                 finalImageR=None,
                 dispL=None,
                 dispR=None,
                 forwardDispChangeL=None,
                 forwardDispChangeR=None,
                 backwardDispChangeL=None,
                 backwardDispChangeR=None,
                 forwardFlowL=None,
                 forwardFlowR=None,
                 backwardFlowL=None,
                 backwardFlowR=None
                 ):
        self._first = first
        self._last = last
        self._cleanImageL = cleanImageL
        self._cleanImageR = cleanImageR
        self._finalImageL = finalImageL
        self._finalImageR = finalImageR
        self._dispL = dispL
        self._dispR = dispR
        self._forwardDispChangeL = forwardDispChangeL
        self._forwardDispChangeR = forwardDispChangeR
        self._backwardDispChangeL = backwardDispChangeL
        self._backwardDispChangeR = backwardDispChangeR
        self._forwardFlowL = forwardFlowL
        self._forwardFlowR = forwardFlowR
        self._backwardFlowL = backwardFlowL
        self._backwardFlowR = backwardFlowR
        self._format = format

    def first(self): return self._first
    def last(self): return self._last

    def _check(self, name, x):
        #if x is None:
        #    raise Exception('Error: %s is None' % name)
        return x

    def cleanImageL(self): return self._check('cleanImageL', self._cleanImageL)
    def finalImageL(self): return self._check('finalImageL', self._finalImageL)

    def cleanImageR(self): return self._check('cleanImageR', self._cleanImageR)
    def finalImageR(self): return self._check('finalImageR', self._finalImageR)

    def dispL(self): return self._check('dispL', self._dispL)
    def dispR(self): return self._check('dispR', self._dispR)
    def forwardDispChangeL(self): return self._check('forwardDispChangeL', self._forwardDispChangeL)
    def forwardDispChangeR(self): return self._check('forwardDispChangeR', self._forwardDispChangeR)
    def backwardDispChangeL(self): return self._check('backwardDispChangeL', self._backwardDispChangeL)
    def backwardDispChangeR(self): return self._check('backwardDispChangeR', self._backwardDispChangeR)
    def forwardFlowL(self): return self._check('forwardFlowL', self._forwardFlowL)
    def forwardFlowR(self): return self._check('forwardFlowR', self._forwardFlowR)
    def backwardFlowL(self): return self._check('backwardFlowL', self._backwardFlowL)
    def backwardFlowR(self): return self._check('backwardFlowR', self._backwardFlowR)

    def hasCleanImageL(self): return self._cleanImageL is not None
    def hasDispL(self): return self._dispL is not None
    def hasDispR(self): return self._dispR is not None
    def hasForwardDispChangeL(self): return self._forwardDispChangeL is not None
    def hasForwardDispChangeR(self): return self._forwardDispChangeR is not None
    def hasBackwardDispChangeL(self): return self._backwardDispChangeL is not None
    def hasBackwardDispChangeR(self): return self._backwardDispChangeR is not None
    def hasForwardFlowL(self): return self._forwardFlowL is not None
    def hasForwardFlowR(self): return self._forwardFlowR is not None
    def hasBackwardFlowL(self): return self._backwardFlowL is not None
    def hasBackwardFlowR(self): return self._backwardFlowR is not None

def computeHistograms(resolution, subpath, collectionName, clips, skipIfExists=False, overwrite=True, numBins=5000, maxValue=1000):
    dataPath = '/misc/lmbraid17/sceneflownet/common/data/4_bin-db'
    savePath = '%s/hists/%s/%s' % (dataPath, resolution, subpath)
    saveFile = '%s/%s.npz' % (savePath, collectionName)

    completePath = os.path.dirname(saveFile)
    os.system('mkdir -p %s' % completePath)

    print 'savePath', savePath
    print 'completePath', completePath
    print 'saveFile', saveFile

    histFlow = np.zeros(numBins)
    histDisp = np.zeros(numBins)
    histDispChange = np.zeros(numBins)
    for clip in clips:
        print 'processing', clip
        for i in range(clip.startFrame(), clip.endFrame()+1):
            frame = clip.frame(i)

            if i%10 == 1:
              print '%d/%d' % (i - clip.startFrame() + 1, clip.endFrame() - clip.startFrame() + 1)

            # flow histogram
            if frame.hasForwardFlowL():
                values = tb.readFlow(frame.forwardFlowL())
                mag = np.power(np.sum(np.power(values, 2), axis=2), 0.5)
                curr_hist, _ = np.histogram(mag, bins=numBins, range=(0,maxValue))
                histFlow += curr_hist

            # disparity histogram
            if frame.hasDispL():
                values = tb.readDisparity(frame.dispL())
                mag = np.abs(values)
                curr_hist, _  = np.histogram(mag, bins=numBins, range=(0,maxValue))
                histDisp += curr_hist

            # disparity change histogram
            if frame.hasBackwardDispChangeL():
                values  = tb.readDisparity(frame.forwardDispChangeL())
                mag = np.abs(values)
                curr_hist, _  = np.histogram(mag, bins=numBins, range=(0,maxValue))
                histDispChange += curr_hist

    np.savez(saveFile, histFlow=histFlow, histDisp=histDisp, histDispChange=histDispChange)

def createBinDB(resolution, subpath, collectionName, clips, entryTypes=None, downsample=1, skipIfExists=False, overwrite=True):
    if entryTypes is None:
        entryTypes = ['cleanImageL', 'finalImageL', 'cleanImageR', 'finalImageR', 'forwardFlowL', 'forwardFlowR', 'backwardFlowL', 'backwardFlowR', 'dispL', 'dispR', 'forwardDispChangeL', 'forwardDispChangeR', 'backwardDispChangeL', 'backwardDispChangeR']

    def entryProto(name, type, channels):
        txt = ''
        txt += '        entry_format: {\n'
        txt += '            name: "%s"\n' % name
        txt += '            data_encoding: %s \n' % type
        txt += '            width: %d\n' % width
        txt += '            height: %d\n' % height
        txt += '            channels: %d\n' % channels
        txt += '        }\n'
        return txt

    def fileProto(filename, entries):
        txt = ''
        txt += 'file {\n'
        txt += '    filename: "%s"\n' % filename
        txt += '    content: {\n'
        for entry in entries:
            txt += entry
        txt += '    }\n'
        txt += '}\n'
        txt += '\n'
        return txt

    def writeCleanImage(path):
        if path is 0 or path is None:
            np.array([0], dtype=np.int32).tofile(fCleanImage)
            np.zeros((3,height,width)).astype(np.uint8).tofile(fCleanImage)
        else:
            np.array([1], dtype=np.int32).tofile(fCleanImage)
            Entry('image',path=path).data(downsample=downsample).tofile(fCleanImage)

    def writeFinalImage(path):
        if path is 0 or path is None:
            np.array([0], dtype=np.int32).tofile(fFinalImage)
            np.zeros((3,height,width)).astype(np.uint8).tofile(fFinalImage)
        else:
            np.array([1], dtype=np.int32).tofile(fFinalImage)
            Entry('image',path=path).data(downsample=downsample).tofile(fFinalImage)

    def writeFlow(path):
        if path is 0 or path is None:
            np.array([0], dtype=np.int32).tofile(fFlow)
            np.zeros((2,height,width)).astype(np.uint16).tofile(fFlow)
        else:
            np.array([1], dtype=np.int32).tofile(fFlow)
            Entry('flow',path=path).data(downsample=downsample).tofile(fFlow)

    def writeDispL(path):
        if path is 0 or path is None:
            np.array([0], dtype=np.int32).tofile(fDisp)
            np.zeros((1,height,width)).astype(np.uint16).tofile(fDisp)
        else:
            np.array([1], dtype=np.int32).tofile(fDisp)
            Entry('leftdisparity',path=path).data(downsample=downsample).tofile(fDisp)

    def writeDispR(path):
        if path is 0 or path is None:
            np.array([0], dtype=np.int32).tofile(fDisp)
            np.zeros((1,height,width)).astype(np.uint16).tofile(fDisp)
        else:
            np.array([1], dtype=np.int32).tofile(fDisp)
            Entry('rightdisparity',path=path).data(downsample=downsample).tofile(fDisp)

    def writeDispChangeL(path):
        if path is 0 or path is None:
            np.array([0], dtype=np.int32).tofile(fDispChange)
            np.zeros((1,height,width)).astype(np.uint16).tofile(fDispChange)
        else:
            np.array([1], dtype=np.int32).tofile(fDispChange)
            Entry('leftdisparitychange',path=path).data(downsample=downsample).tofile(fDispChange)

    def writeDispChangeR(path):
        if path is 0 or path is None:
            np.array([0], dtype=np.int32).tofile(fDispChange)
            np.zeros((1,height,width)).astype(np.uint16).tofile(fDispChange)
        else:
            np.array([1], dtype=np.int32).tofile(fDispChange)
            Entry('rightdisparitychange',path=path).data(downsample=downsample).tofile(fDispChange)

    dataPath = '/misc/lmbraid17/sceneflownet/common/data/4_bin-db'

    path = '%s/%s/%s/%s' % (dataPath, resolution, subpath, collectionName)
    if os.path.exists(path):
        if not overwrite and (skipIfExists or not tb.queryYesNo('The collection %s already exists. Do you want to delete it and create a new one?' % collectionName)):
            return
        else:
            os.system('rm -rf %s' % path)

    print path
    os.system('mkdir -p %s' % path)

    num = 0
    width = -1
    height = -1
    slice_points = []
    print 'collection', collectionName, path
    for clip in clips:
        frame = clip.frame(clip.startFrame())
        if frame.hasCleanImageL():            
          image = misc.imread(frame.cleanImageL())
        else: 
          image = misc.imread(frame.finalImageL())
        if width==-1 and height==-1:
            width = image.shape[1]
            height = image.shape[0]
        if width!= image.shape[1] or height!=image.shape[0]:
            raise Exception('clips in collection %s have different resolutions' % collection)

        length = clip.endFrame() - clip.startFrame() + 1
        print 'adding clip %d %d (%d)' % (clip.startFrame(), clip.endFrame(), length)
        num += length
        print 'slice_point %d' % num
        slice_points.append(num)

    txt = ''
    txt += 'num: %d\n' % num
    for point in slice_points[:-1]:
        txt += 'slice_before: %d\n' % point
    txt += '\n'

    cleanImageDefs = []
    if 'cleanImageL' in entryTypes: cleanImageDefs.append(entryProto('cleanImageL','UINT8',3))
    if 'cleanImageR' in entryTypes: cleanImageDefs.append(entryProto('cleanImageR','UINT8',3))
    if len(cleanImageDefs): txt += fileProto('image_clean.bin', cleanImageDefs)

    finalImageDefs = []
    if 'finalImageL' in entryTypes: finalImageDefs.append(entryProto('finalImageL','UINT8',3))
    if 'finalImageR' in entryTypes: finalImageDefs.append(entryProto('finalImageR','UINT8',3))
    if len(finalImageDefs): txt += fileProto('image_final.bin', finalImageDefs)

    flowDefs = []
    if 'forwardFlowL' in entryTypes: flowDefs.append(entryProto('forwardFlowL', 'FIXED16DIV32', 2))
    if 'forwardFlowR' in entryTypes: flowDefs.append(entryProto('forwardFlowR', 'FIXED16DIV32', 2))
    if 'backwardFlowL' in entryTypes: flowDefs.append(entryProto('backwardFlowL', 'FIXED16DIV32', 2))
    if 'backwardFlowR' in entryTypes: flowDefs.append(entryProto('backwardFlowR', 'FIXED16DIV32', 2))
    if len(flowDefs): txt += fileProto('flow.bin', flowDefs)

    disparityDefs = []
    if 'dispL' in entryTypes: disparityDefs.append(entryProto('dispL', 'FIXED16DIV32', 1))
    if 'dispR' in entryTypes: disparityDefs.append(entryProto('dispR', 'FIXED16DIV32', 1))
    if len(disparityDefs): txt += fileProto('disparity.bin', disparityDefs)


    disparityChangeDefs = []
    if 'forwardDispChangeL' in entryTypes: disparityChangeDefs.append(entryProto('forwardDispChangeL', 'FIXED16DIV32', 1))
    if 'forwardDispChangeR' in entryTypes: disparityChangeDefs.append(entryProto('forwardDispChangeR', 'FIXED16DIV32', 1))
    if 'backwardDispChangeL' in entryTypes: disparityChangeDefs.append(entryProto('backwardDispChangeL', 'FIXED16DIV32', 1))
    if 'backwardDispChangeR' in entryTypes: disparityChangeDefs.append(entryProto('backwardDispChangeR', 'FIXED16DIV32', 1))
    if len(disparityChangeDefs): txt += fileProto('disparity_change.bin', disparityChangeDefs)
    txt += '\n'

    open('%s/index.prototxt' % path, 'w').write(txt)

    if len(cleanImageDefs): fCleanImage = open('%s/image_clean.bin' % path, 'w')
    if len(finalImageDefs): fFinalImage = open('%s/image_final.bin' % path, 'w')
    if len(flowDefs): fFlow = open('%s/flow.bin' % path, 'w')
    if len(disparityDefs): fDisp = open('%s/disparity.bin' % path, 'w')
    if len(disparityChangeDefs): fDispChange = open('%s/disparity_change.bin' % path, 'w')

    for clip in clips:
        for i in range(clip.startFrame(), clip.endFrame()+1):
            frame = clip.frame(i)
            print '%d:' % i

            if 'cleanImageL' in entryTypes:
                print 'cleanImageL:', frame.cleanImageL()
                writeCleanImage(frame.cleanImageL())
            if 'cleanImageR' in entryTypes:
                print 'cleanImageR:', frame.cleanImageR()
                writeCleanImage(frame.cleanImageR())

            if 'finalImageL' in entryTypes:
                print 'finalImageL:', frame.finalImageL()
                writeFinalImage(frame.finalImageL())
            if 'finalImageR' in entryTypes:
                print 'finalImageR:', frame.finalImageR()
                writeFinalImage(frame.finalImageR())

            if not frame.last():
                if 'forwardFlowL' in entryTypes:
                    print 'forwardFlowL:', frame.forwardFlowL()
                    writeFlow(frame.forwardFlowL())
                if 'forwardFlowR' in entryTypes:
                    print 'forwardFlowR:', frame.forwardFlowR()
                    writeFlow(frame.forwardFlowR())
                if 'forwardDispChangeL' in entryTypes:
                    print 'forwardDispChangeL', frame.forwardDispChangeL()
                    writeDispChangeL(frame.forwardDispChangeL())
                if 'forwardDispChangeR' in entryTypes:
                    print 'forwardDispChangeR', frame.forwardDispChangeR()
                    writeDispChangeR(frame.forwardDispChangeR())
            else:
                if 'forwardFlowL' in entryTypes:
                    print 'forwardFlowL:', 0
                    writeFlow(0)
                if 'forwardFlowR' in entryTypes:
                    print 'forwardFlowR:', 0
                    writeFlow(0)
                if 'forwardDispChangeL' in entryTypes:
                    print 'forwardDispChangeL', 0
                    writeDispChangeL(0)
                if 'forwardDispChangeL' in entryTypes:
                    print 'forwardDispChangeL', 0
                    writeDispChangeR(0)

            if not frame.first():
                if 'backwardFlowL' in entryTypes:
                    print 'backwardFlowL:', frame.backwardFlowL()
                    writeFlow(frame.backwardFlowL())
                if 'backwardFlowR' in entryTypes:
                    print 'backwardFlowR:', frame.backwardFlowR()
                    writeFlow(frame.backwardFlowR())
                if 'backwardDispChangeL' in entryTypes:
                    print 'backwardDispChangeL', frame.backwardDispChangeL()
                    writeDispChangeL(frame.backwardDispChangeL())
                if 'backwardDispChangeR' in entryTypes:
                    print 'backwardDispChangeR', frame.backwardDispChangeR()
                    writeDispChangeR(frame.backwardDispChangeR())
            else:
                if 'backwardFlowL' in entryTypes:
                    print 'backwardFlowL:', 0
                    writeFlow(0)
                if 'backwardFlowR' in entryTypes:
                    print 'backwardFlowR:', 0
                    writeFlow(0)
                if 'backwardDispChangeL' in entryTypes:
                    print 'backwardDispChangeL', 0
                    writeDispChangeL(0)
                if 'backwardDispChangeL' in entryTypes:
                    print 'backwardDispChangeL', 0
                    writeDispChangeR(0)

            if 'dispL' in entryTypes:
                print 'dispL:', frame.dispL()
                writeDispL(frame.dispL())
            if 'dispR' in entryTypes:
                print 'dispR:', frame.dispR()
                writeDispR(frame.dispR())

class BlenderFrame:
    def __init__(self, path, number):
        self._path = path
        self._number = number

    def number(self):
        return self._number

    def first(self):
        return not os.path.isfile('%s/converted/Cleanpass_%04d_L.png' % (self._path, self._number - 1))

    def last(self):
        return not os.path.isfile('%s/converted/Cleanpass_%04d_L.png' % (self._path, self._number + 1))

    def _checkFilename(self, path):
        if not os.path.isfile(path):
            raise Exception('File %s does not exist' % path)
        return path

    def cleanImageL(self):
        return self._checkFilename('%s/converted/Cleanpass_%04d_L.png' % (self._path, self._number))

    def finalImageL(self):
        return self._checkFilename('%s/raw/Finalpass_%04d_L.png' % (self._path, self._number))

    def cleanImageR(self):
        return self._checkFilename('%s/converted/Cleanpass_%04d_R.png' % (self._path, self._number))

    def finalImageR(self):
        return self._checkFilename('%s/raw/Finalpass_%04d_R.png' % (self._path, self._number))

    def dispL(self):
        return self._checkFilename('%s/converted/Disparity_%04d_L.pfm' % (self._path, self._number))

    def dispR(self):
        return self._checkFilename('%s/converted/Disparity_%04d_R.pfm' % (self._path, self._number))

    def forwardDispChangeL(self):
        return self._checkFilename('%s/converted/DisparityChangeIntoFuture_%04d_L.pfm' % (self._path, self._number))

    def forwardDispChangeR(self):
        return self._checkFilename('%s/converted/DisparityChangeIntoFuture_%04d_R.pfm' % (self._path, self._number))

    def backwardDispChangeL(self):
        return self._checkFilename('%s/converted/DisparityChangeIntoPast_%04d_L.pfm' % (self._path, self._number))

    def backwardDispChangeR(self):
        return self._checkFilename('%s/converted/DisparityChangeIntoPast_%04d_R.pfm' % (self._path, self._number))

    def forwardFlowL(self):
        return self._checkFilename('%s/converted/OpticalFlowIntoFuture_%04d_L.pfm' % (self._path, self._number))

    def forwardFlowR(self):
        return self._checkFilename('%s/converted/OpticalFlowIntoFuture_%04d_R.pfm' % (self._path, self._number))

    def backwardFlowL(self):
        return self._checkFilename('%s/converted/OpticalFlowIntoPast_%04d_L.pfm' % (self._path, self._number))

    def backwardFlowR(self):
        return self._checkFilename('%s/converted/OpticalFlowIntoPast_%04d_R.pfm' % (self._path, self._number))

    def hasCleanImageL(self): return True
    def hasDispL(self): return True
    def hasDispR(self): return True
    def hasForwardDispChangeL(self): return not self.last()
    def hasForwardDispChangeR(self): return not self.last()
    def hasBackwardDispChangeL(self): return not self.first()
    def hasBackwardDispChangeR(self): return not self.first()
    def hasForwardFlowL(self): return not self.last()
    def hasForwardFlowR(self): return not self.last()
    def hasBackwardFlowL(self): return not self.first()
    def hasBackwardFlowR(self): return not self.first()

class Clip:
    def __init__(self, movie, setting, collection, startFrame=None, endFrame=None):
        self._movie = movie
        self._setting = setting
        self._collection = collection
        self._startFrame = startFrame
        self._endFrame = endFrame
        self._frames = []

    def addFrame(self, frame): self._frames.append(frame)
    def frame(self, number):  return self._frames[number]

    def movie(self): return self._movie
    def setting(self): return self._setting
    def collection(self): return self._collection

    def startFrame(self):
        if self._startFrame is None: return 0
        else:                        return self._startFrame
    def endFrame(self):
        if self._endFrame is None:   return len(self._frames) - 1
        else:                        return self._endFrame

    def __repr__(self): return 'clip(%s)' % self._collection

class BlenderClip:
    dataPath = '/misc/lmbraid17/sceneflownet/common/data/2_blender-out/lowres'
    #dataPath = '/mnt/sceneflownet_archive/sceneflownet/common/data/2_blender-out/lowres'

    def __init__(self, line):
        parts = line.split()
        self._movie = parts[0]
        self._setting = parts[1]
        self._startFrame = int(parts[2])
        self._endFrame = int(parts[3])
        self._collection = parts[4]
        self._path = '%s/%s/%s' % (self.dataPath, self._movie, self._setting)

    def frame(self, number): return BlenderFrame(self._path, number)

    def movie(self): return self._movie
    def setting(self): return self._setting
    def collection(self): return self._collection

    def startFrame(self): return self._startFrame
    def endFrame(self): return self._endFrame

    def __repr__(self): return 'clip(%s)' % self._collection

def readCollections(files):
    list = []
    for file in files:
        for l in open(file).readlines():
            if l.strip() == '' or l.startswith('#'): continue
            list.append(BlenderClip(l))

        map = OrderedDict()
        for clip in list:
            name = clip.movie()+'/'+clip.collection()
            if name not in map: map[name] = []
            map[name].append(clip)

    return map

def makeFlowCheck(resolution, rendertype, clip):
    print clip.path()
    checkPath = clip.path() + '/flow_check'
    os.system('mkdir -p %s' % checkPath)
    for i in range(clip.startFrame(), clip.endFrame()):
        print 'frame %d' % i
        frame = clip.frame(i, rendertype)

        Image0 = misc.imread(frame.ImageL())
        Image1 = misc.imread(frame.ImageL(+1))
        flow  = tb.readPFM(frame.forwardFlowL())[0][:,:,0:2]

        warped = tb.flowWarp(Image1, flow)

        h = Image0.shape[0]
        w = Image0.shape[1]
        ImageCheck = np.zeros((2*h, 2*w, 3)).astype(np.uint8)
        ImageCheck[0:h, 0:w, :] = Image0
        ImageCheck[0:h, w:2*w, :] = Image1
        ImageCheck[h:2*h, 0:w, :] = warped

        misc.imsave('%s/%s.png' % (checkPath, os.path.basename(frame.ImageL())), ImageCheck)

class Entry:
    def __init__(self, type, name=None, path=None):
        self._type = type
        self._path = path
        self._name = name

    def channels(self):
        if   self._type == "image": return 3
        elif self._type == "flow": return 2
        elif self._type == "leftdisparity": return 1
        elif self._type == "rightdisparity": return 1
        raise Exception('unhandled data type')

    def name(self):
        return self._name

    def caffeType(self):
        if   self._type == "image": return 'UINT8'
        elif self._type == "flow": return 'FIXED16DIV32'
        elif self._type == "leftdisparity": return 'FIXED16DIV32'
        elif self._type == "rightdisparity": return 'FIXED16DIV32'

    def data(self, downsample = 1):
        if self._type == "image":
            Image = misc.imread(self._path)[:, :, 0:3]
            if downsample != 1: Image = tb.downsampleImage(Image, downsample)
            return Image[..., np.r_[2, 1, 0]].transpose((2, 0, 1))
        elif self._type == "flow":
            flow = tb.readFlow(self._path)
            flow = flow[:, :, 0:2]
            if downsample != 1: flow = OpticalFlow.downsampleMedian(flow, downsample)
            return (flow.transpose((2, 0, 1)) * 32.0).astype(np.int16)
        elif self._type == "leftdisparity":
            disparity = tb.readDisparity(self._path)
            disparity *= -1
            if downsample != 1: raise Exception("no downsampling implemented for disparity")
            return (disparity * 32.0).astype(np.int16)
        elif self._type == "rightdisparity":
            disparity = tb.readDisparity(self._path)
            if downsample != 1: raise Exception("no downsampling implemented for disparity")
            return (disparity * 32.0).astype(np.int16)
        elif self._type == "leftdisparitychange":
            disparityChange = tb.readDisparity(self._path)
            disparityChange *= -1
            if downsample != 1: raise Exception("no downsampling implemented for disparity")
            return (disparityChange * 32.0).astype(np.int16)
        elif self._type == "rightdisparitychange":
            disparityChange = tb.readDisparity(self._path)
            if downsample != 1: raise Exception("no downsampling implemented for disparity")
            return (disparityChange * 32.0).astype(np.int16)
        else:
            raise Exception('unhandled data type')

    def dims(self, downsample = 1):
        if self._type == "image":
            Image = misc.imread(self._path)[:, :, 0:3]
            if downsample != 1: Image = tb.downsampleImage(Image, downsample)
            return (Image.shape[1], Image.shape[0])
        raise Exception('dimensions not implemented for datatype')

























































#
# def createLMDB(resolution, rendertype, type, name, clipList, entitySize, downsample, skipIfExists=False, overwrite=True):
#     dataPath = '/misc/lmbraid17/sceneflownet/common/data/3_caffe-db'
#
#     path = '%s/%s/%s/%s/%s' % (dataPath, resolution, rendertype, type, name)
#
#     if os.path.exists(path):
#         if not overwrite and (skipIfExists or not tb.queryYesNo('The LMDB %s/%s already exists. Do you want to delete it and create a new one?' % (type, name))):
#             return
#         else:
#             os.system('rm -rf %s' % path)
#
#     os.system('mkdir -p %s' % path)
#
#     if type == 'flow':
#         dataset = FlowDataset(name, rendertype, path, entitySize)
#         dataset.addClips(clipList)
#         dataset.generateLMDB(downsample)
#     elif type == 'disparity':
#         dataset = DispDataset(name, rendertype, path, entitySize)
#         dataset.addClips(clipList)
#         dataset.generateLMDB(downsample)
#     elif type == 'sceneflow':
#         dataset = SceneflowDataset(name, rendertype, path, entitySize)
#         dataset.addClips(clipList)
#         dataset.generateLMDB(downsample)
#     else:
#         raise Exception('unsupported')



#
# class DatasetEntry:
#     def __init__(self, id):
#         self._layers = []
#         self._id = id
#
#     def addLayer(self, layer):
#         self._layers.append(layer)
#
#     def layers(self):
#         return self._layers
#
#     def dims(self, downsample=1):
#         return self._layers[0].dims(downsample)
#
#     def channels(self):
#         c = 0
#         for layer in self._layers:
#             c += layer.channels()
#         return c
#
#     def data(self, downsample=1):
#         d = ''
#         for layer in self._layers:
#             d += layer.data(downsample).tostring()
#         return d
#
#     def label(self):
#         return self._id
#
# class DatasetInfo:
#     def __init__(self):
#         self._fields = {}
#         self._layers = []
#
#     def set(self, key, value):
#         self._fields[key] = value
#
#     def addLayers(self, entry):
#         i = 0
#         for layer in entry.layers():
#             self._layers.append('layer%d/%s: %dx%s' % (i, layer.name(), layer.channels(), layer.caffeType()))
#             i += 1
#
#     def writeTo(self, path):
#         f = open(path, 'w')
#         for key, value in self._fields.iteritems():
#             f.write('%s: %s\n' % (key, value))
#         for layer in self._layers:
#             f.write('%s\n' % layer)
#
# class Dataset:
#     def __init__(self, name, rendertype, path):
#         self._name = name
#         self._path = path
#         self._rendertype = rendertype
#         self._entries = []
#
#     def addClips(self, clips):
#         for clip in clips:
#             self.addClip(clip)
#
#     def generateLMDB(self, downsample = 1):
#         db = tb.openLMDB('%s/lmdb' % self._path, True)
#         db_counter = 0
#
#         entries = np.random.permutation(self._entries)
#
#         txn = db.begin(write=True)
#         labelMap = open('%s/map.txt' % self._path, 'w')
#         first = True
#         for entry in entries:
#             dims = entry.dims(downsample)
#             if first:
#                 info = DatasetInfo()
#                 info.set('width', dims[0])
#                 info.set('height', dims[1])
#                 info.set('channels', entry.channels())
#                 info.set('lmdb', '%s/lmdb' % self._path)
#                 info.addLayers(entry)
#                 info.writeTo('%s/info.txt' % self._path)
#                 first = False
#
#             datum = caffe.proto.caffe_pb2.Datum()
#             datum.height = dims[1]
#             datum.width = dims[0]
#             datum.channels = entry.channels()
#             datum.label = 0
#             datum.data = entry.data(downsample)
#
#             label = '%08d%s' % (db_counter, entry.label())
#             print len(datum.SerializeToString())
#             txn.put(label, datum.SerializeToString())
#             labelMap.write('%s\n' % label)
#             print 'added entry', label
#             db_counter += 1
#
#             # if db_counter % 100 == 0:
#             #     labelMap.close()
#             #     txn.commit()
#             #     return
#
#             if db_counter % 25 == 0:
#                 print 'submitted %d entries' % db_counter
#                 txn.commit()
#                 txn = db.begin(write=True)
#
#         print 'submitted %d entries' % db_counter
#         labelMap.close()
#         txn.commit()
#
# class FlowDataset(Dataset):
#     def __init__(self, name, rendertype, path, entitySize):
#         Dataset.__init__(self, name, rendertype, path)
#         self._entitySize = entitySize
#
#     def addClip(self, clip):
#         def idFor(k):
#             return clip.frame(k, self._rendertype).ImageL().replace('/misc/lmbraid17/sceneflownet/common', '').replace('/', '_').replace('.', '_')
#
#         print 'adding scene %s' % clip.name()
#
#         for k in range(clip.startFrame(), clip.endFrame() - self._entitySize + 1):
#             leftForward = DatasetEntry('%s_forward' % idFor(k))
#             for i in range(0, self._entitySize):   leftForward.addLayer(DatasetLayer('image', 'image%d' % i, clip.frame(k + i, self._rendertype).ImageL()))
#             for i in range(0, self._entitySize-1): leftForward.addLayer(DatasetLayer('flow',  'flow%d%d' % (i, i+1), clip.frame(k + i, self._rendertype).forwardFlowL()))
#             self._entries.append(leftForward)
#
#             rightForward = DatasetEntry('%s_forward' % idFor(k))
#             for i in range(0, self._entitySize):   rightForward.addLayer(DatasetLayer('image', 'image%d' % i, clip.frame(k + i, self._rendertype).ImageR()))
#             for i in range(0, self._entitySize-1): rightForward.addLayer(DatasetLayer('flow', 'flow%d%d' % (i, i+1), clip.frame(k + i, self._rendertype).forwardFlowR()))
#             self._entries.append(rightForward)
#
#             leftBackward = DatasetEntry('%s_backward' % idFor(k))
#             for i in reversed(range(0, self._entitySize)): leftBackward.addLayer(DatasetLayer('image', 'image%d' % (self._entitySize - i - 1), clip.frame(k + i, self._rendertype).ImageL()))
#             for i in reversed(range(1, self._entitySize)): leftBackward.addLayer(DatasetLayer('flow', 'flow%d%d' % (self._entitySize - i - 1, self._entitySize - i - 2), clip.frame(k + i, self._rendertype).backwardFlowL()))
#             self._entries.append(leftBackward)
#
#             rightBackward = DatasetEntry('%s_backward' % idFor(k))
#             for i in reversed(range(0, self._entitySize)): rightBackward.addLayer(DatasetLayer('image', 'image%d' % (self._entitySize - i - 1), clip.frame(k + i, self._rendertype).ImageR()))
#             for i in reversed(range(1, self._entitySize)): rightBackward.addLayer(DatasetLayer('flow', 'flow%d%d' % (self._entitySize - i - 1, self._entitySize - i - 2), clip.frame(k + i, self._rendertype).backwardFlowR()))
#             self._entries.append(rightBackward)
#
# class DispDataset(Dataset):
#     def __init__(self, name, rendertype, path, entitySize):
#         Dataset.__init__(self, name, rendertype, path)
#         self._entitySize = entitySize
#
#     def addClip(self, clip):
#         def idFor(k):
#             return clip.frame(k, self._rendertype).ImageL().replace('/misc/lmbraid17/sceneflownet/common', '').replace('/', '_').replace('.', '_')
#
#         print 'adding scene %s' % clip.name()
#
#         for k in range(clip.startFrame(), clip.endFrame() - self._entitySize + 1):
#             leftEntry = DatasetEntry('%s' % idFor(k))
#             for i in range(0, self._entitySize):
#                 leftEntry.addLayer(DatasetLayer('image', 'image%dL' % i, clip.frame(k + i, self._rendertype).ImageL()))
#                 leftEntry.addLayer(DatasetLayer('image', 'image%dR' % i, clip.frame(k + i, self._rendertype).ImageR()))
#             for i in range(0, self._entitySize):
#                 leftEntry.addLayer(DatasetLayer('leftdisparity', 'disp%d' % i, clip.frame(k + i, self._rendertype).disparityL()))
#             self._entries.append(leftEntry)
#
#             rightEntry = DatasetEntry('%s' % idFor(k))
#             for i in range(0, self._entitySize):
#                 rightEntry.addLayer(DatasetLayer('image', 'image%dL' % i, clip.frame(k + i, self._rendertype).ImageR()))
#                 rightEntry.addLayer(DatasetLayer('image', 'image%dR' % i, clip.frame(k + i, self._rendertype).ImageL()))
#             for i in range(0, self._entitySize):
#                 rightEntry.addLayer(DatasetLayer('rightdisparity', 'disp%d' % i, clip.frame(k + i, self._rendertype).disparityR()))
#             self._entries.append(rightEntry)
#
# class SceneflowDataset(Dataset):
#     def __init__(self, name, rendertype, path, entitySize):
#         Dataset.__init__(self, name, rendertype, path)
#         if entitySize < 2:
#             raise Exception('scene flow entity size must be at least 2')
#         self._entitySize = entitySize
#
#     def addClip(self, clip):
#         def idFor(k):
#             return clip.frame(k, self._rendertype).ImageL().replace('/misc/lmbraid17/sceneflownet/common', '').replace('/', '_').replace('.', '_')
#
#         print 'adding scene %s' % clip.name()
#
#         for k in range(clip.startFrame(), clip.endFrame() - self._entitySize + 1):
#             forwardEntry = DatasetEntry('%s' % idFor(k))
#             for i in range(0, self._entitySize):
#                 forwardEntry.addLayer(DatasetLayer('image', 'image%dL' % i, clip.frame(k + i, self._rendertype).ImageL()))
#                 forwardEntry.addLayer(DatasetLayer('image', 'image%dR' % i, clip.frame(k + i, self._rendertype).ImageR()))
#             for i in range(0, self._entitySize):
#                 forwardEntry.addLayer(DatasetLayer('leftdisparity', 'disp%d' % i, clip.frame(k + i, self._rendertype).disparityL()))
#             for i in range(0, self._entitySize-1): forwardEntry.addLayer(DatasetLayer('flow',  'flow%d%dL' % (i, i+1), clip.frame(k + i, self._rendertype).forwardFlowL()))
#             for i in range(0, self._entitySize-1): forwardEntry.addLayer(DatasetLayer('flow',  'flow%d%dR' % (i, i+1), clip.frame(k + i, self._rendertype).forwardFlowR()))
#             self._entries.append(forwardEntry)
#
#             backwardEntry = DatasetEntry('%s' % idFor(k))
#             for i in reversed(range(0, self._entitySize)):
#                 backwardEntry.addLayer(DatasetLayer('image', 'image%dL' % (self._entitySize - i - 1), clip.frame(k + i, self._rendertype).ImageL()))
#                 backwardEntry.addLayer(DatasetLayer('image', 'image%dR' % (self._entitySize - i - 1), clip.frame(k + i, self._rendertype).ImageR()))
#             for i in reversed(range(0, self._entitySize)):
#                 backwardEntry.addLayer(DatasetLayer('leftdisparity', 'disp%d' % (self._entitySize - i - 1), clip.frame(k + i, self._rendertype).disparityL()))
#             for i in reversed(range(1, self._entitySize)): backwardEntry.addLayer(DatasetLayer('flow',  'flow%d%dL' % (self._entitySize - i - 1, self._entitySize - i - 2), clip.frame(k + i, self._rendertype).backwardFlowL()))
#             for i in reversed(range(1, self._entitySize)): backwardEntry.addLayer(DatasetLayer('flow',  'flow%d%dR' % (self._entitySize - i - 1, self._entitySize - i - 2), clip.frame(k + i, self._rendertype).backwardFlowR()))
#             self._entries.append(backwardEntry)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#     # def _readTest(self, tuples):
#     #     tuples = np.random.permutation(tuples)
#     #
#     #     i = 0
#     #     d = 0
#     #     for t in tuples:
#     #         start_time = time.time()
#     #         Image1 = misc.imread(t[1])[:, :, 0:3]
#     #         Image2 = misc.imread(t[2])[:, :, 0:3]
#     #         flow, scale = tb.readPFM(t[3])
#     #         d += (time.time()-start_time)
#     #         i += 1
#     #         if i % 10 == 0:
#     #             d /= i
#     #             print 'access time: %fms' % (d*1000)
#     #             d = 0
#     #             i = 0
#
#
#
#
#
#
#
#
#
#
#
#
#     # def _generateLMDB(self, tuples, downsample):
#     #     db = tb.openLMDB('%s/lmdb' % self._path, True)
#     #     db_counter = 0
#     #
#     #     tuples = np.random.permutation(tuples)
#     #
#     #     txn = db.begin(write=True)
#     #     labelMap = open('%s/map.txt' % self._path, 'w')
#     #     for t in tuples:
#     #
#     #         label = t[0]
#     #         Image1 = tb.downsampleImage(misc.imread(t[1])[:, :, 0:3], downsample)
#     #         Image2 = tb.downsampleImage(misc.imread(t[2])[:, :, 0:3], downsample)
#     #         flow, scale = tb.readPFM(t[3])
#     #         flow = OpticalFlow.downsampleMedian(flow[:, :, 0:2], downsample)
#     #
#     #         datum = caffe.proto.caffe_pb2.Datum()
#     #         datum.height = Image1.shape[0]
#     #         datum.width = Image1.shape[1]
#     #         datum.channels = 3 + 3 + 2 + 1
#     #         datum.label = 0
#     #
#     #         datum.data = ''
#     #         datum.data += Image1[..., np.r_[2, 1, 0]].transpose((2, 0, 1)).tostring()
#     #         datum.data += Image2[..., np.r_[2, 1, 0]].transpose((2, 0, 1)).tostring()
#     #         datum.data += (flow.transpose((2, 0, 1)) * 32.0).astype(np.int16).tostring()
#     #         datum.data += np.zeros((datum.width * datum.height - 1) / 8 + 1).astype(np.uint8).tostring()
#     #
#     #         label = '%08d%s' % (db_counter, label)
#     #         txn.put(label, datum.SerializeToString())
#     #         labelMap.write('%s\n' % label)
#     #         print 'added entry', label
#     #         db_counter += 1
#     #
#     #         if db_counter % 100 == 0:
#     #             print 'submitted %d entries' % db_counter
#     #             txn.commit()
#     #             txn = db.begin(write=True)
#     #
#     #     print 'submitted %d entries' % db_counter
#     #     labelMap.close()
#     #     txn.commit()
#     #
#     #



# def createBinDB(resolution, rendertype, collectionName, clips, downsample, skipIfExists=False, overwrite=True):
#     def entryProto(name, type, channels):
#         txt = ''
#         txt += '        entry_format: {\n'
#         txt += '            name: "%s"\n' % name
#         txt += '            data_encoding: %s \n' % type
#         txt += '            width: %d\n' % width
#         txt += '            height: %d\n' % height
#         txt += '            channels: %d\n' % channels
#         txt += '        }\n'
#         return txt
#
#     def fileProto(filename, entries):
#         txt = ''
#         txt += 'file {\n'
#         txt += '    filename: "%s"\n' % filename
#         txt += '    content: {\n'
#         for entry in entries:
#             txt += entry
#         txt += '    }\n'
#         txt += '}\n'
#         txt += '\n'
#         return txt
#
#     def writeImage(path):
#         if path is 0:
#             np.array([0], dtype=np.int32).tofile(fImage)
#             np.zeros((3,height,width)).astype(np.uint8).tofile(fImage)
#         else:
#             np.array([1], dtype=np.int32).tofile(fImage)
#             Entry('image',path=path).data(downsample=downsample).tofile(fImage)
#
#     def writeFlow(path):
#         if path is 0:
#             np.array([0], dtype=np.int32).tofile(fFlow)
#             np.zeros((2,height,width)).astype(np.uint16).tofile(fFlow)
#         else:
#             np.array([1], dtype=np.int32).tofile(fFlow)
#             Entry('flow',path=path).data(downsample=downsample).tofile(fFlow)
#
#     def writeDispL(path):
#         if path is 0:
#             np.array([0], dtype=np.int32).tofile(fDisp)
#             np.zeros((1,height,width)).astype(np.uint16).tofile(fDisp)
#         else:
#             np.array([1], dtype=np.int32).tofile(fDisp)
#             Entry('leftdisparity',path=path).data(downsample=downsample).tofile(fDisp)
#
#     def writeDispR(path):
#         if path is 0:
#             np.array([0], dtype=np.int32).tofile(fDisp)
#             np.zeros((1,height,width)).astype(np.uint16).tofile(fDisp)
#         else:
#             np.array([1], dtype=np.int32).tofile(fDisp)
#             Entry('rightdisparity',path=path).data(downsample=downsample).tofile(fDisp)
#
#     def writeDispChangeL(path):
#         if path is 0:
#             np.array([0], dtype=np.int32).tofile(fDispChange)
#             np.zeros((1,height,width)).astype(np.uint16).tofile(fDispChange)
#         else:
#             np.array([1], dtype=np.int32).tofile(fDispChange)
#             Entry('leftdisparitychange',path=path).data(downsample=downsample).tofile(fDispChange)
#
#     def writeDispChangeR(path):
#         if path is 0:
#             np.array([0], dtype=np.int32).tofile(fDispChange)
#             np.zeros((1,height,width)).astype(np.uint16).tofile(fDispChange)
#         else:
#             np.array([1], dtype=np.int32).tofile(fDispChange)
#             Entry('rightdisparitychange',path=path).data(downsample=downsample).tofile(fDispChange)
#
#     dataPath = '/misc/lmbraid17/sceneflownet/common/data/4_bin-db'
#
#     path = '%s/%s/%s/%s' % (dataPath, resolution, rendertype, collectionName)
#     if os.path.exists(path):
#         if not overwrite and (skipIfExists or not tb.queryYesNo('The collection %s already exists. Do you want to delete it and create a new one?' % collectionName)):
#             return
#         else:
#             os.system('rm -rf %s' % path)
#
#     os.system('mkdir -p %s' % path)
#
#     num = 0
#     width = -1
#     height = -1
#     slice_points = []
#     print 'collection', collectionName
#     for clip in clips:
#         Image = misc.imread(clip.frame(clip.startFrame(), rendertype).ImageL())
#         if width==-1 and height==-1:
#             width = Image.shape[1]
#             height = Image.shape[0]
#         if width!= Image.shape[1] or height!=Image.shape[0]:
#             raise Exception('clips in collection %s have different resolutions' % collection)
#
#         len = clip.endFrame() - clip.startFrame() + 1
#         print 'adding clip %d %d (%d)' % (clip.startFrame(), clip.endFrame(), len)
#         num += len
#         print 'slice_point %d' % num
#         slice_points.append(num)
#
#     txt = ''
#     txt += 'num: %d\n' % num
#     for point in slice_points[:-1]:
#         txt += 'slice_before: %d\n' % point
#     txt += '\n'
#     txt += fileProto('image.bin', [entryProto('imageL','UINT8',3), entryProto('imageR','UINT8',3)])
#     txt += fileProto('flow.bin', [entryProto('forwardFlowL', 'FIXED16DIV32', 2), entryProto('forwardFlowR', 'FIXED16DIV32', 2), entryProto('backwardFlowL', 'FIXED16DIV32', 2), entryProto('backwardFlowR', 'FIXED16DIV32', 2)])
#     txt += fileProto('disparity.bin', [entryProto('dispL', 'FIXED16DIV32', 1), entryProto('dispR', 'FIXED16DIV32', 1)])
#     txt += fileProto('disparity_change.bin', [entryProto('forwardDispChangeL', 'FIXED16DIV32', 1), entryProto('forwardDispChangeR', 'FIXED16DIV32', 1), entryProto('backwardDispChangeL', 'FIXED16DIV32', 1), entryProto('backwardDispChangeR', 'FIXED16DIV32', 1)])
#     txt += '\n'
#
#     open('%s/index.prototxt' % path, 'w').write(txt)
#
#     fImage = open('%s/image.bin' % path, 'w')
#     fFlow = open('%s/flow.bin' % path, 'w')
#     fDisp = open('%s/disparity.bin' % path, 'w')
#     fDispChange = open('%s/disparity_change.bin' % path, 'w')
#
#     for clip in clips:
#         for i in range(clip.startFrame(), clip.endFrame()+1):
#             frame = clip.frame(i, rendertype)
#             print '%d:' % i
#
#             writeImage(frame.ImageL()); print 'imageL:', frame.ImageL()
#             writeImage(frame.ImageR()); print 'imageR:', frame.ImageR()
#
#             if not frame.last():
#                 writeFlow(frame.forwardFlowL()); print 'forwardFlowL:', frame.forwardFlowL()
#                 writeFlow(frame.forwardFlowR()); print 'forwardFlowR:', frame.forwardFlowR()
#                 writeDispChangeL(frame.forwardDispChangeL()); print 'forwardDispChangeL', frame.forwardDispChangeL()
#                 writeDispChangeR(frame.forwardDispChangeR()); print 'forwardDispChangeR', frame.forwardDispChangeR()
#             else:
#                 writeFlow(0); print 'forwardFlowL:', 0
#                 writeFlow(0); print 'forwardFlowR:', 0
#                 writeDispChangeL(0); print 'forwardDispChangeL', 0
#                 writeDispChangeR(0); print 'forwardDispChangeL', 0
#
#             if not frame.first():
#                 writeFlow(frame.backwardFlowL()); print 'backwardFlowL:', frame.backwardFlowL()
#                 writeFlow(frame.backwardFlowR()); print 'backwardFlowR:', frame.backwardFlowR()
#                 writeDispChangeL(frame.backwardDispChangeL()); print 'backwardDispChangeL', frame.backwardDispChangeL()
#                 writeDispChangeR(frame.backwardDispChangeR()); print 'backwardDispChangeR', frame.backwardDispChangeR()
#             else:
#                 writeFlow(0); print 'backwardFlowL:', 0
#                 writeFlow(0); print 'backwardFlowR:', 0
#                 writeDispChangeL(0); print 'backwardDispChangeL', 0
#                 writeDispChangeR(0); print 'backwardDispChangeL', 0
#
#             writeDispL(frame.disparityL()); print 'dispL:', frame.disparityL()
#             writeDispR(frame.disparityR()); print 'dispR:', frame.disparityR()
#
