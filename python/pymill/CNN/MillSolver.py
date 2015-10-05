#!/usr/bin/python

from __future__ import division
import caffe
from pylab import *
import numpy as np
import os
import sqlite3 as lite
import json
from datetime import datetime, date
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import threading


class MillSolver(object):
    def __init__(self, solver_def, solver_state=None, weights=None, gpus=[0], log_dir='log', log_db_prefix='log_db'):
        """
        Acts as a driver for training with smart logging.
        Logs will be stored to a MySQL database.
        All paths can be set relative to the location of the solver prototxt.

        :param solver_def:      prototxt that defines the solver
        :param solver_state:    optional: a .solverstate file from which to resume training \  NEVER SET THESE TWO
        :param weights:         optional: a .caffemodel file from which to begin finetuning /   AT THE SAME TIME
        :param gpus:            optional: a list of GPU IDs to use for (multi-)GPU training
                                if set to None, caffe will operate in CPU mode
        :param log_dir:         optional: will log into this directory under solver.prototxt
        :param log_db_prefix:   prefix for both SQLite db names and table names

        The following parameters should to be set in the solver prototxt file:
        log_interval            log per this number of iterations (simple log) [default = 20]
        viz_interval            log visualization per this number of iterations (net blobs snapshot) [default = 100]
        test_iter:              The number of iterations for each test net.
        """
        if not os.path.isabs(solver_def):
            if not os.path.isfile(os.path.join(os.getcwd(), solver_def)):
                os.chdir('..')
                solver_def = os.path.join(os.getcwd(), solver_def)
            else:
                solver_def = os.path.join(os.getcwd(), solver_def)

        self.solver_dir = solver_def[:solver_def.rfind('/')]
        os.chdir(self.solver_dir)
        self.log_dir = os.path.join(self.solver_dir, log_dir)
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        self.logprint("Logging to {}".format(self.log_dir))
        self.log_db_prefix = log_db_prefix  # used for db name and as a prefix for tables
        self.solver_param = caffe_pb2.SolverParameter()
        text_format.Merge(open(solver_def).read(), self.solver_param)

        # read params from solver definition
        self.iterations = self.solver_param.max_iter
        self.log_interval = self.solver_param.log_interval
        self.viz_interval = self.solver_param.viz_interval
        self.test_interval = self.solver_param.test_interval

        # make solver param for net fail safe
        if not os.path.isabs(self.solver_param.net):
            self.solver_param.net = os.path.join(self.solver_dir, self.solver_param.net)
            if not os.path.isfile(self.solver_param.net):
                raise Exception('could not find net definition from solver prototxt!')

        self.gpus = gpus
        if gpus:
            self.solver_param.device_id = gpus[0]
            caffe.set_device(gpus[0])
            caffe.set_mode_gpu()
            caffe.set_solver_count(len(gpus))

        self.solver = caffe.get_solver_from_string(self.solver_param.SerializeToString())

        if solver_state:
            # check if this file is in the current (or parent) directory or if the solver path needs to be prepended
            if not os.path.isfile(os.path.join('..', solver_state)):
                if not os.path.isfile(solver_state):
                    solver_state = os.path.join(self.solver_dir, solver_state)
                    if not os.path.isfile(solver_state):
                        raise Exception('could not find solver state specified!')
            else:
                solver_state = os.path.join('..', solver_state)
            self.solver.restore(solver_state)

            if weights:
                raise Exception(
                    'should not specify both solverstate and caffemodel! Preference will be given to solverstate.')

        if weights and not solver_state:
            self.solver.net.copy_from(weights)

        self.sync = None
        self.viz_thread = None
        self.log_thread = None
        self.test_input = None
        self.test_out_blobs = None
        self.test_start_layer = None
        self.iteration = 0

    def run_solver(self):
        """
        Kicks off (multi-)GPU training
        """

        if len(self.gpus) > 1:
            self.sync = caffe.P2PSync(self.solver, self.solver_param.SerializeToString())
            self.sync.set_pysolver(self)
            self.sync.set_callback_iteration(1)
            self.sync.run(self.gpus)

        else:  # fallback: classical single GPU training
            for i in range(self.iterations):
                self.solver.step(1)
                self.iteration = self.solver.iter
                self.stats_and_log()

    def stats_and_log(self):
        iteration = self.iteration

        if iteration % self.log_interval == 0:
            # extract loss
            self.logprint("#### DEBUG: start logging")
            loss_log = dict()
            for output in self.solver.net.outputs:
                loss_log[output] = float(self.solver.net.blobs[output].data)

            # extract percentiles
            self.logprint("#### DEBUG: start calculating blob percentiles")
            blob_percentiles_log = dict()
            for blob in self.solver.net.blobs:
                if len(self.solver.net.blobs[blob].data.shape) > 1:
                    blob_percentiles_log[blob] = self.get_percentiles(self.solver.net.blobs[blob])

            self.logprint("#### DEBUG: start calculating weight percentiles")
            weight_percentiles_log = dict()
            for blob in self.solver.net.params:
                if len(self.solver.net.params[blob][0].data.shape) > 1:
                    weight_percentiles_log[blob + '-w'] = self.get_percentiles(self.solver.net.params[blob][0])
                    weight_percentiles_log[blob + '-b'] = self.get_percentiles(self.solver.net.params[blob][1])

            # extract learning rate
            try:
                lr_log = self.solver.getLearningRate()
            except:
                raise Exception('solver does not support extracting learning rate!')

            # write to DB
            self.logprint("#### DEBUG: start logging thread --> DB")
            self.log_thread = threading.Thread(target=self.write_out_log, args=(iteration, lr_log, loss_log, blob_percentiles_log, weight_percentiles_log,))
            self.log_thread.setDaemon(True)
            self.log_thread.start()

        # visualization log
        if self.viz_interval and iteration % self.viz_interval == 0:
            self.viz_thread = threading.Thread(target=self.write_out_viz, args=(iteration, self.solver.net.blobs,))
            self.viz_thread.setDaemon(True)
            self.viz_thread.start()

        # run test log if necessary
        if self.test_interval and iteration % self.test_interval == 0:
            test_loss_log = dict()
            # extract learning rate
            try:
                lr_log = self.solver.getLearningRate()
            except:
                raise Exception('solver does not support extracting learning rate!')
            for tn in range(len(self.solver.test_nets)):
                # logprint("== Test net #{} (iteration {}):".format(tn, solver.iter-log_per))
                for outp in self.solver.test_nets[tn].outputs:
                    test_loss_log["tn{}-{}".format(tn, outp)] = float(self.solver.test_nets[tn].blobs[outp].data)
                    # logprint("==   {} = {}".format(outp, float(solver.test_nets[tn].blobs[outp].data)))
            self.write_out_log(iteration, lr_log, test_loss_log, lr_log, lr_log, phase="TEST")

            if self.test_input:
                for input in self.test_input:
                    self.solver.net.blobs[input] = self.test_input[input]
                test_output = self.solver.net.forward(start=self.test_start_layer, blobs=self.test_out_blobs)
                self.write_out_viz(iteration, test_output, test=True)

    def callback_gradients(self):
        if not self.solver.iter > self.iterations:
            self.sync.set_callback_iteration(self.solver.iter + 1)
            self.stats_and_log()

    def step_debug(self):
        for i in range(1000):
            self.solver.step(10)
            for tn in range(len(self.solver.test_nets)):
                self.logprint("== Test net #{} (iteration {}):".format(tn, self.solver.iter))
                for outp in self.solver.test_nets[tn].outputs:
                    self.logprint("==   {} = {}".format(outp, float(self.solver.test_nets[tn].blobs[outp].data)))

    def get_percentiles(self, blob):
        """
        Calculates the percentiles for each channel of each data/diff blob and each data/diff layer param
        :param blob: Caffe blob
        :return: list consisting of the percentiles for the data and diff blobs, respectively
        """
        blob_data = blob.data.swapaxes(0, 1)  # squeeze channel to first index s.th.
        blob_diff = blob.diff.swapaxes(0, 1)  # percentiles can be computed more efficiently
        data_percentiles = list(np.percentile(blob_data.flatten(), [0, 15.87, 50, 84.13, 100]))  # JSON serialization
        diff_percentiles = list(np.percentile(blob_diff.flatten(), [0, 15.87, 50, 84.13, 100]))  # can't deal w/ numpy

        return [data_percentiles, diff_percentiles]

    def write_out_viz(self, iteration, blobs, test=False):
        con = None
        suffix = "test_" if test else ""
        table_name = "{}_{}viz".format(self.log_db_prefix, suffix)
        try:
            con, cur = self.get_db_connection(viz=True, timeout=5 * 60)
            cur.execute(
                "create table if not exists {} (ID INT PRIMARY KEY, name TEXT, blob BLOB)".format(
                    table_name))
            cur.execute('begin')
            cmd = '''INSERT OR REPLACE INTO {} VALUES(?, ?, ?);'''.format(table_name)
            if test:
                # housekeeping: delete keys with iteration number >= current iteration number
                cur.execute(
                    '''DELETE FROM {} WHERE name = 'test-output-blobs' AND ID >= {};'''.format(table_name, iteration))
                cur.execute(cmd,
                            [iteration, "test-output-blobs", lite.Binary(json.dumps(blobs, default=self.json_default))])
                con.commit()
            else:
                for idx, blob in enumerate(blobs):
                    cur.execute(cmd, [idx, blob, lite.Binary(blobs[blob].data)])
                    con.commit()
                self.logprint(
                    "logged {} blobs for iteration {} to {}/{}_viz.db/{}".format(len(blobs), iteration, self.log_dir,
                                                                                 self.log_db_prefix, table_name))

        except lite.Error, e:
            raise Exception('Error %s:' % e.args[0])
            # sys.exit(1)

        finally:
            if con:
                con.close()
        self.logprint(" // DEBUG // VIZ thread END")

    def write_out_log(self, iteration, lr_log, loss_log, blob_percentiles_log, weight_percentiles_log,
                      phase="TRAIN"):
        """
        Writes log contents to a SQLite database. Dicts are JSON-serialized.

        :param solver_dir:
        :param iteration:
        :param lr_log:
        :param loss_log:
        :param blob_percentiles_log:
        :param weight_percentiles_log:
        :param testing_results:         set this to true for logging TESTING results, as opposed to training results
        :return:
        """
        con = None
        try:
            con, cur = self.get_db_connection()
            cur.execute(
                "create table if not exists {}(Iteration INT, LearningRate NUMERIC, Losses BLOB, BlobPercentiles BLOB, WeightPercentiles BLOB, timestamp TEXT, Phase TEXT)".format(
                    self.log_db_prefix))

            # housekeeping: delete keys with iteration number >= current iteration number
            cur.execute('''DELETE FROM {} WHERE Phase = '{}' AND Iteration >= {};'''.format(self.log_db_prefix, phase,
                                                                                            iteration))
            con.commit()

            sql = '''INSERT OR REPLACE INTO {} VALUES(?, ?, ?, ?, ?, ?, ?);'''.format(self.log_db_prefix)
            con.execute(sql, [iteration,
                              lr_log,
                              lite.Binary(json.dumps(loss_log)),
                              lite.Binary(json.dumps(blob_percentiles_log)),
                              lite.Binary(json.dumps(weight_percentiles_log)),
                              datetime.now(),
                              phase])
            con.commit()
            self.logprint(
                "wrote {} iteration {} (lr={}) to {}/{}.db".format(phase, iteration, lr_log, self.log_dir,
                                                                   self.log_db_prefix))

        except lite.Error, e:
            raise Exception('Error %s:' % e.args[0])
            # sys.exit(1)

        finally:
            if con:
                con.close()

    def print_db(self, count=None, viz=False):
        sort_by = "ID" if viz else "Iteration"
        con = None
        try:
            # con = lite.connect(os.path.join(self.log_dir, "{}{}.db".format(self.log_db_prefix, suffix)))
            con, cur = self.get_db_connection(viz=viz)
            cur.execute("SELECT * FROM {}{} ORDER BY {} ASC".format(self.log_db_prefix, suffix, sort_by))
            data = cur.fetchall()
            if count:
                max_idx = min(len(data), count)
            else:
                max_idx = len(data)
            headers = list(map(lambda x: x[0], cur.description))
            for s in data[:max_idx]:
                tmp = ""
                for idx, field in enumerate(headers):
                    tmp += "{}: {}\t".format(field, np.array(s[idx]))
                print(tmp + "\n")

        except lite.Error, e:
            raise Exception('Error %s:' % e.args[0])
            # sys.exit(1)

        finally:
            if con:
                con.close()

    def logprint(self, string):
        """
        convenience function for console logging with prefix and timestamp
        :param string: stuff to log
        """
        print("VIZLOG [{}]: {}".format(datetime.now(), string))

    def get_db_connection(self, viz=False, timeout=10):
        suffix = "_viz" if viz else ""
        try:
            con = lite.connect(os.path.join(self.log_dir, "{}{}.db".format(self.log_db_prefix, suffix)),
                               timeout=timeout)
            return con, con.cursor()

        except lite.Error, e:
            raise Exception('Error %s:' % e.args[0])

    def set_test_input(self, input, start_layer):
        """
        set a test input to be forwarded through the net every so many iterations
        :param input: dict where keys are input blob names and values are blob ndarrays.
        :param start_layer:
        :return:
        """
        assert type(input) == dict or type(input) == type(self.solver.net.blobs), "Input must be of type (ordered) dict"
        assert type(input[input.keys()[0]]) == type(
            self.solver.net.blobs[self.solver.net.blobs.keys()[0]]), "Values must be of type caffe._caffe.Blob"
        self.test_input = input

        assert type(
            start_layer) == str, "start_layer must be a string with the name of the layer to which the test blob is input"
        assert start_layer in self.solver.net._layer_names, "Could not find the start layer in the net definition. Typo?"
        self.test_start_layer = start_layer

    def set_test_outputs(self, blobs):
        """
        set the names of the blobs that should be captured for the test blob forward pass
        :param blobs: list of blob names
        :return:
        """
        assert type(blobs) == list, "Output test blobs must be provided as a list like ['conv1', 'ip2']."
        self.test_out_blobs = blobs

    def json_default(self, obj):
        """
        :param obj: makes a list of numpy arrays --> simpler JSON serialization
        :return: the object as a list
        """
        try:
            return obj.tolist()
        except:
            raise TypeError('Not serializable')
