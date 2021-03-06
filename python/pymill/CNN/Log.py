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
from collections import OrderedDict
import datetime


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


class Plot:
    def __init__(self, title):
        # This makes the figure squeezeable
        plt.figure(figsize=(1,1))
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(8, 6, forward=True)

        plt.grid(color='0.65')
        plt.title(title)

        self._legend = []

    def plotList(self, label, list, style):
        x = np.zeros((len(list), 1))
        y = np.zeros((len(list), 1))

        i = 0
        for pair in list:
            x[i] = pair[0]
            y[i] = pair[1]
            i += 1

        plt.plot(x, y, color=style.color(), linestyle=style.lineStyle(), linewidth=style.lineWidth())
        self._legend.append(label)

    def plotSmoothedList(self, label, list, style):
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
        self._legend.append(label)

    def finish(self):
        fontP = FontProperties()
        fontP.set_size('small')
        plt.legend(self._legend, prop=fontP, bbox_to_anchor=(1.12,1.0))
        plt.show()

class Log:
    def __init__(self, networkName, filename):
        lines = open(filename,'r').readlines()

        iter = -1

        self._networkName = networkName
        self._lines = []
        self._measures = {}
        self._measureList = []

        def appendMeasure(name, iter, value):
            if name not in self._measures:
                 self._measures[name]=[]
                 self._measureList.append(name)
            self._measures[name].append((iter,float(value)))

        for l in lines:
            if ']' not in l:
                self._lines.append((iter, l))
                continue

            msg = l.split(']')[1].strip()

            if msg.startswith('Iteration'):
                match = re.compile('Iteration ([0-9]+)').match(msg)
                if match:
                    iter = int(match.group(1))

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
                match = re.compile('Test net output #[a-zA-Z0-9_-]+: ([a-zA-Z0-9_-]+) = (([0-9]|\.)+)').match(msg)
                if match:
                    name = match.group(1)
                    value = match.group(2)
                    appendMeasure('test_'+name, iter, value)

            self._lines.append((iter, l))

        self._measureList = tb.unique(self._measureList)

    def eta(self, max_iter):
        from dateutil.parser import parse
        itertime = 0
        lastiter = -1
        lasttime = -1
        times = []
        for iter, line in self._lines:
            if iter!=lastiter:
                match = re.compile('.([0-9]{2})([0-9]{2}) ([0-9]{2}:[0-9]{2}:[0-9]{2}).*').match(line)
                if match:
                    fulldate = '00%02d-%02d-%02d %s' % (15, int(match.group(1)), int(match.group(2)), match.group(3))
                    time = parse(fulldate)

                    if lasttime != -1:
                        niters = iter - lastiter
                        diff = time - lasttime
                        times.append(diff.seconds/float(niters))


                    lasttime = time

                lastiter = iter

        time = np.mean(times[-100:])

        remaining = max_iter - lastiter
        time*= remaining

        delta = datetime.timedelta(seconds=time)

        return (datetime.datetime.now() + delta).strftime('%a %d.%m.%Y %H:%M')

    def displayBlobSummary(self):
        sizes = OrderedDict()
        assignments = []
        for iter, line in self._lines:
            if ']' in line:
                msg = line.split(']')[1].strip()
                match = re.compile('([a-zA-Z0-9_-]+) -> ([a-zA-Z0-9_-]+).*').match(msg)
                if match:
                    assignments.append(match.group(2))
                else:
                    match = re.compile('Top shape: ([0-9]+) ([0-9]+) ([0-9]+) ([0-9]+).*').match(msg)
                    if match:
                        value = assignments.pop(0)
                        sizes[value]=(int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4)))
                        #print '%50s: %d %d %d %d' % (value, int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4)))
                    match = re.compile('Top shape: \(([0-9]+)\).*').match(msg)
                    if match:
                        value = assignments.pop(0)
                        sizes[value]=int(match.group(1))
                        #print '%50s: %d ' % (value, int(match.group(1)))

        for ass in assignments:
            sizes[ass] = (-1, -1, -1, -1)

        def printMapping(layer, s, dir):
            if isinstance(s, int):
                print '%70s %s (%4d)' % (blob, dir,  s)
            else:
                print '%70s %s (%4d %4d %4d %4d)' % (blob, dir, s[0], s[1], s[2], s[3])

        layer = None
        for iter, line in self._lines:
            if ']' in line:
                msg = line.split(']')[1].strip()

                match = re.compile('Creating Layer ([a-zA-Z0-9_-]+)').match(msg)
                if match:
                    layer = match.group(1).strip()
                    print '%70s ------------------------' % ('Layer %s' % layer)

                match = re.compile('([a-zA-Z0-9_-]+) -> ([a-zA-Z0-9_-]+).*').match(msg)
                if match:
                    blob = match.group(2)
                    printMapping(layer, sizes[blob], '->')

                match = re.compile('([a-zA-Z0-9_-]+) <- ([a-zA-Z0-9_-]+).*').match(msg)
                if match:
                    blob = match.group(2)
                    printMapping(layer, sizes[blob], '<-')



    def networkName(self): return self._networkName
    def measures(self): return self._measures
    def measureNames(self): return self._measureList

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

    def plot(self, select=''):
        measureList = []
        if select == '':
            measureList = self._measureList
        else:
            selections = select.split(',')
            for selection in selections:
                measureList += tb.wildcardMatch(self._measureList, selection)
        measureList = tb.unique(measureList)

        plot = Plot("loss/accuracy for %s" % self._networkName)

        def plotMeasure(name,label,color):
            plot.plotList(label, self._measures[name], color)
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

        plot.finish()


    def plotlr(self):
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

        plot = Plot("learning rate for %s" % self._networkName)

        if len(lrs):
            plot.plotList('LR', lrs, tb.PlotStyle('r-'))
            plt.ylim((0, maxLr * 1.1))

        plot.finish()

    @staticmethod
    def plotComparison(names, logs):
        plot = Plot("loss/accuracy comparison")

        styles = tb.styleList()
        for log in logs:
            style = styles.pop(0)
            #lineStyles = tb.lineStyleCycle()
            for name in names:
                label = '%s for %s' %(name,log.networkName())
                #style.setLineStyle(lineStyles.get())
                if name in log.measures():
                    plot.plotList(label, log.measures()[name], style)

        plot.finish()