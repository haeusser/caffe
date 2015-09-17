#!/usr/bin/python

import cython

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np

# declare the interface to the C code
cdef extern void c_downsample(float* xyflow, float* down, int m, int n, int f)

@cython.boundscheck(False)
@cython.wraparound(False)

def downsample(np.ndarray[float, ndim=3, mode="c"] input not None, int f):
    cdef int m, n

    if f == 1: 
        return input 

    m, n = input.shape[0], input.shape[1]

    cdef np.ndarray[float, ndim=3, mode="c"] output 
    output = np.zeros((m/f, n/f, 2)).astype(np.float32)

    c_downsample (&input[0,0,0], &output[0,0,0], m, n, f)

    return output

