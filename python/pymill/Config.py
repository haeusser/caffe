#!/usr/bin/python

from pymill.Toolbox import PlotStyle

plotMeasureStyles = {
    'train_loss':       ('Loss',        PlotStyle('r-')),
    'test_loss':        ('Test loss',   PlotStyle('g-')),
    'test_f_measure':   ('F-measure',   PlotStyle('g--')),
    'test_precision':   ('Precision',   PlotStyle('y:')),
    'test_recall':      ('Recall',      PlotStyle('c:')),
    'test_psnr':        ('PSNR',        PlotStyle('g--')),
}

#prototmpInclude = '/home/ilge/data/caffe/include'
prototmpInclude = '/misc/lmbraid17/sceneflownet/common/prototmp'

