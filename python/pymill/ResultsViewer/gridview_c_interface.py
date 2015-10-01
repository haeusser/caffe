
from numpy.ctypeslib import ndpointer
import ctypes
import numpy as np
import os
import gridview


libfile = os.path.join(os.path.dirname(__file__),
                       'c_backend',
                       'FlowVisualization.so')

lib = ctypes.cdll.LoadLibrary(libfile)


cColorFlow = lib.ColorFlow
cColorFlow.restype = None
cColorFlow.argtypes = [ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                       ndpointer(ctypes.c_ubyte, flags='C_CONTIGUOUS'),
                       ctypes.c_int,
                       ctypes.c_float]

def ColorFlow(raw_flow_data, scale=1.0):
  shape = raw_flow_data.shape
  output = np.zeros((shape[0],shape[1],3), dtype=np.uint8)
  cColorFlow(raw_flow_data, output, shape[0]*shape[1], scale)
  return output



cColorFlow2 = lib.ColorFlow2
cColorFlow2.restype = None
cColorFlow2.argtypes = [ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                        ndpointer(ctypes.c_ubyte, flags='C_CONTIGUOUS'),
                        ctypes.c_int,
                        ctypes.c_float]

def ColorFlow2(raw_flow_data, scale=1.0):
  shape = raw_flow_data.shape
  output = np.zeros((shape[0],shape[1],3), dtype=np.uint8)
  cColorFlow2(raw_flow_data, output, shape[0]*shape[1], scale)
  return output



def Flow(style, raw_flow_data, scale):
  return [ColorFlow, ColorFlow2][style](raw_flow_data, scale)

