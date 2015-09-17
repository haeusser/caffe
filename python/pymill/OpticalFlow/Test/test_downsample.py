#!/usr/bin/python

import Toolbox as tb
import OpticalFlow.downsample as downsample
import numpy as np

flow = tb.readFlow('flow.flo')

flow_ds2 = downsample(flow, 2)
flow_ds4 = downsample(flow, 4)

tb.writeFlow('ds2.flo', flow_ds2)
tb.writeFlow('ds4.flo', flow_ds4)