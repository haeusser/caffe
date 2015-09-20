#!/usr/bin/python

"""
run this file from the test directory
"""

import sys

sys.path.append('/misc/lmbraid17/sceneflownet/haeusserp/hackathon-caffe2/python/pymill/CNN')  # avoid pymill __init__ for now
sys.path.append('/home/haeusser/libs/hackathon-caffe2/python/pymill/CNN')  # avoid pymill __init__ for now
try:
    from CNN import MillSolver as ms
except:
    import MillSolver as ms

import unittest
import os
from unittest import TestSuite


class TestMS(unittest.TestCase):
    """
    def test_train_and_resume(self):
        log_db_prefix = 'log_db'
        ms.run_solver(solver_def='/misc/lmbraid17/sceneflownet/common/nets/haeusserp/mnist/training/solver.prototxt',
               solver_state=None, iterations=30, log_per=2, viz_per_logs=0, test_per_logs=0, solver_type='AdamSolver')

        self.assertEqual(True, os.path.isfile('/misc/lmbraid17/sceneflownet/common/nets/haeusserp/mnist/training/mnist_iter_25.solverstate'))

        ms.run_solver(solver_def='/misc/lmbraid17/sceneflownet/common/nets/haeusserp/mnist/training/solver.prototxt',
               solver_state='/misc/lmbraid17/sceneflownet/common/nets/haeusserp/mnist/training/mnist_iter_25.solverstate',
                      iterations=30, log_per=2, viz_per_logs=2, test_per_logs=2, solver_type='AdamSolver')

        con = None
        try:
            import sqlite3 as lite
            con = lite.connect("{}/{}.db".format('/misc/lmbraid17/sceneflownet/common/nets/haeusserp/mnist/training', log_db_prefix))
            cur = con.cursor()
            cur.execute("SELECT Iteration, Losses, timestamp FROM {} ORDER BY Iteration DESC".format(log_db_prefix))
            iteration, losses, timestamp = cur.fetchone()
        except lite.Error, e:
            raise NameError('Error %s:' % e.args[0])
            # sys.exit(1)
        finally:
            if con:
                con.close()

        self.assertEqual(iteration, 53)

    def test_mnist_solver_quick(self):
        log_db_prefix = 'log_db'
        iterations = 2
        ms.run_solver(solver_def='/misc/lmbraid17/sceneflownet/common/nets/haeusserp/mnist/training/solver.prototxt',
               solver_state=None, iterations=iterations, log_per=1, viz_per_logs=2, test_per_logs=2, solver_type=None)
        con = None
        try:
            import sqlite3 as lite
            con = lite.connect("{}/{}.db".format('/misc/lmbraid17/sceneflownet/common/nets/haeusserp/mnist/training', log_db_prefix))
            cur = con.cursor()
            cur.execute("SELECT Iteration, Losses, timestamp FROM {} ORDER BY Iteration DESC".format(log_db_prefix))
            iteration, losses, timestamp = cur.fetchone()
        except lite.Error, e:
            raise NameError('Error %s:' % e.args[0])
            # sys.exit(1)
        finally:
            if con:
                con.close()

        self.assertEqual(iteration, iterations)
    """

    def test_mnist_solver_long(self):
        log_db_prefix = 'log_db'
        solver = ms.MillSolver(solver_def='mnist/solver.prototxt')
        solver.run_solver()
        con, cur = ms.MillSolver.get_db_connection(solver)
        cur.execute("SELECT Iteration FROM {} ORDER BY Iteration DESC".format(log_db_prefix))
        iteration = cur.fetchone()
        self.assertEqual(iteration[0], 100)

    def test_mnist_solver_long_finetune(self):
        log_db_prefix = 'log_db'
        solver = ms.MillSolver(solver_def='mnist/solver.prototxt',
                               solver_state='mnist/lenet_iter_50.solverstate')
        solver.run_solver()
        con, cur = ms.MillSolver.get_db_connection(solver)
        cur.execute("SELECT Iteration FROM {} ORDER BY Iteration DESC".format(log_db_prefix))
        iteration = cur.fetchone()
        self.assertEqual(iteration[0], 150)

        # clean up unnecessary snapshots
        files = os.listdir('.')
        for file in files:
            if 'caffemodel' in file or 'solverstate' in file:
                os.remove(file)

    def test_mnist_solver_multi_gpu(self):
        log_db_prefix = 'log_db'
        solver = ms.MillSolver(solver_def='mnist/solver.prototxt', gpus=[0, 1])
        solver.run_solver()
        con, cur = ms.MillSolver.get_db_connection(solver)
        cur.execute("SELECT Iteration FROM {} ORDER BY Iteration DESC".format(log_db_prefix))
        iteration = cur.fetchone()
        self.assertEqual(iteration[0], 100)

    def test_test_blob(self):
        solver = ms.MillSolver(solver_def='mnist/solver.prototxt')
        solver.solver.step(1)
        test_blob = dict({'data': solver.solver.net.blobs['data']})
        test_outputs = ['ip2']
        solver.set_test_input(test_blob)
        solver.set_test_outputs(test_outputs)
        solver.run_solver()



if __name__ == '__main__':
    unittest.main()
