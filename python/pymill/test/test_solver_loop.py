#!/usr/bin/python

import unittest
import os
from unittest import TestSuite
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import timeit

class TestSolverLoop(unittest.TestCase):
    def run_solver_step(self):
        solver_def='solver.prototxt'
        solver_param = caffe_pb2.SolverParameter()
        text_format.Merge(open(solver_def).read(), solver_param)
        solver = caffe.get_solver_from_string(solver_param.SerializeToString())
        solver.step(50)

    def run_solver_loop(self):
        solver_def='solver.prototxt'
        solver_param = caffe_pb2.SolverParameter()
        text_format.Merge(open(solver_def).read(), solver_param)
        solver = caffe.get_solver_from_string(solver_param.SerializeToString())
        for i in range(50):
            solver.step(1)

    def wrapper(self, func, *args, **kwargs):
        def wrapped():
            return func(*args, **kwargs)
        return wrapped

    def test_solver_loop_mnist(self):
        os.chdir('mnist')
        global wrapped_step
        wrapped_step = self.wrapper(self.run_solver_step)
        global wrapped_loop
        wrapped_loop = self.wrapper(self.run_solver_loop)

        timing_step = timeit.timeit(wrapped_step, number=5)
        timing_loop = timeit.timeit(wrapped_loop, number=5)

        print("5x50 steps took:\n\tsteps:\t{} sec\n\tloop:\t{} sec".format(timing_step, timing_loop))
        self.assertAlmostEqual(timing_step, timing_loop, delta=min(timing_loop, timing_step)*0.007)

if __name__ == '__main__':
    unittest.main()
