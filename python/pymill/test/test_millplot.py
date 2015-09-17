#!/usr/bin/python

"""
run this file from the test directory
"""

import sys
sys.path.append('/misc/lmbraid17/sceneflownet/haeusserp/pymill/CNN')  # avoid pymill __init__ for now
sys.path.append('/home/haeusser/libs/pymill/CNN')  # avoid pymill __init__ for now
sys.path.append('/usr/wiss/haeusser/libs/pymill/CNN')  # avoid pymill __init__ for now
try:
    from CNN import MillPlot as mp
except:
    import MillPlot as mp

import unittest
import os
from unittest import TestSuite

class TestMP(unittest.TestCase):

    def test_millplot(self):
        plotter = mp.MillPlot(location='mnist/log/')
        plotter.fetch_range()
        plotter.plot_percentiles()


if __name__ == '__main__':
    unittest.main()
