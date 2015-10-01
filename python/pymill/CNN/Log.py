#!/usr/bin/python

import os
import numpy as np
import re
import matplotlib.pyplot as plt
import pymill.Toolbox as tb
from math import log
from math import pow
from math import sqrt
from matplotlib.font_manager import FontProperties
import matplotlib
from pymill import Config

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
        flat window will produce a moving average smoothing.
    output:
        the smoothed signal

   example:
        t=linspace(-2,2,0.1)
        x=sin(t)+randn(len(t))*0.1
        y=smooth(x)

   see also:
        np.hanning, np.hamming, np.bartlett, np.blackman, np.convolve
        scipy.signal.lfilter

        TODO: the window parameter could be the window itself if an array instead of a string
        NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


class Log:
    _lines = []

    def __init__(self,filename):
        lines = open(filename,'r').readlines()

        iter = -1

        for l in lines:
            if ']' not in l:
                self._lines.append((iter, l))
                continue

            msg = l.split(']')[1].strip()

            if msg.startswith('Iteration'):
                match = re.compile('Iteration ([0-9]+)').match(msg)
                if match:
                    iter = int(match.group(1))

            self._lines.append((iter, l))

    def writeUpTo(self,filename,iteration):
        f = open(filename, 'w')

        for l in self._lines:
            if l[0] <= iteration:
                f.write(l[1])
            else:
                break

    def getAssignment(self, name):
        for l in self._lines:
            if not ']' in l[1]: continue

            iter = l[0]
            msg = l[1].split(']')[1].strip()

            match = re.compile(name+' = (([0-9]|\.)+)').match(msg)
            if match:
                return float(match.group(1))



    def plot(self, networkName, select=''):
        def plotList(list, style):
            x = np.zeros((len(list), 1))
            y = np.zeros((len(list), 1))

            i = 0
            for pair in list:
                x[i] = pair[0]
                y[i] = pair[1]
                i += 1

            plt.plot(x, y, color=style.color(), linestyle=style.lineStyle(), linewidth=style.lineWidth())

        def plotSmoothedList(list, style):
            x = np.zeros((len(list), 1))
            y = np.zeros((len(list), 1))

            i = 0
            for pair in list:
                x[i] = pair[0]
                y[i] = pair[1]
                i += 1

            r = 40
            y = smooth(y.squeeze(),window='hanning', window_len=2 * r + 1)
            y = y[r:len(y) - r]
            x = x.squeeze()
            print x.shape
            print y.shape
            plt.plot(x, y, style)

        measures = {}
        measureList = []

        def appendMeasure(name,iter,value):
            if name not in measures:
                measures[name]=[]
                measureList.append(name)
            measures[name].append((iter,float(value)))

        for l in self._lines:
            iter = l[0]

            if not ']' in l[1]: continue
            msg = l[1].split(']')[1].strip()

            if msg.startswith('Iteration'):
                match = re.compile('Iteration [0-9]+, loss = (([0-9]|\.)+)').match(msg)
                if match:
                    appendMeasure('train_loss', iter, match.group(1))

            if msg.startswith('Test loss'):
                value = re.compile('Test loss: (([0-9]|\.)+)').match(msg).group(1)
                appendMeasure('test_loss', iter, value)

            if msg.startswith('Train net output'):
                match = re.compile('Train net output ..: ([a-zA-Z0-9_-]+) = (([0-9]|\.)+)').match(msg)
                if match:
                    name = match.group(1)
                    value = match.group(2)
                    appendMeasure('train_'+name, iter, value)

            if msg.startswith('Test net output'):
                match = re.compile('Test net output ..: ([a-zA-Z0-9_-]+) = (([0-9]|\.)+)').match(msg)
                if match:
                    name = match.group(1)
                    value = match.group(2)
                    appendMeasure('test_'+name, iter, value)

        tmpMeasureList = measureList
        measureList = []

        if select == '':
            measureList = tmpMeasureList
        else:
            selections = select.split(',')
            for selection in selections:
                measureList += tb.wildcardMatch(tmpMeasureList, selection)

        measureList = tb.unique(measureList)


        # This makes the figure squeezeable
        plt.figure(figsize=(1,1))
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(8, 6, forward=True)

        plt.grid(color='0.65')
        plt.title("Losses and accuracy over iterations for %s" % networkName)

        legend = []

        def plotMeasure(name,label,color):
            plotList(measures[name],color)
            legend.append(label)
            measureList.remove(name)

        for name, (label, style) in Config.plotMeasureStyles.iteritems():
            if name in measureList:
                plotMeasure(name, name, style)

        styles = tb.styleList()
        for name in measureList[:]:
            if name.startswith('test_'):
                plotMeasure(name, name, styles.pop(0))

        for name in measureList[:]:
            plotMeasure(name, name, styles.pop(0))

        fontP = FontProperties()
        fontP.set_size('small')

        plt.legend(legend, prop=fontP, bbox_to_anchor=(1.12,1.0))


    def plotlr(self,name):
        def plotList(list, style):
            x = np.zeros((len(list), 1))
            y = np.zeros((len(list), 1))

            i = 0
            for pair in list:
                x[i] = pair[0]
                y[i] = pair[1]
                i += 1

            plt.plot(x, y, style)

        lrs = []
        maxLr = 0
        for l in self._lines:
            iter = l[0]

            if not ']' in l[1]: continue
            msg = l[1].split(']')[1].strip()

            if msg.startswith('Iteration'):
                match = re.compile('Iteration [0-9]+, lr = (([0-9]|\.|-|e)+)').match(msg)
                if match:
                    lr = float(match.group(1))
                    if lr > maxLr:
                        maxLr = lr
                    lrs.append((iter, lr))


        plt.figure()
        plt.grid(color='0.65')
        plt.title("Learning rate over iterations for %s" % name)

        legend = []

        if len(lrs):
            plotList(lrs,'r-')
            legend.append('LR')

            plt.ylim((0, maxLr * 1.1))


        fontP = FontProperties()
        fontP.set_size('small')
        plt.legend(legend, prop=fontP, bbox_to_anchor=(1.12,1.0))

