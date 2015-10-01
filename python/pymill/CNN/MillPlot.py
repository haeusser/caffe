import sqlite3 as lite
import os
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib import cm


class MillPlot(object):
    def __init__(self, location, log_db_prefix='log_db'):
        """
        Creates a plotter tool that can be used to easily visualize logged data from MillSolver
        :param location: the directory where the log_db.db is located
        :param log_db_prefix: the name of the db and the log table inside (default: 'log_db'
        :return:
        """
        self.location = location
        self.log_db_prefix = log_db_prefix
        self.cache_log_train = None
        self.cache_log_test = None
        self.cache_viz = None
        self.cached_iterations = None
        self.train_outputs = None
        self.test_outputs = None
        self.blob_percentiles_keys = None
        self.weight_percentiles_keys = None
        self.percentile_labels = ['0', '16', '50', '84', '100']
        self.num_percentiles = len(self.percentile_labels)
        self.ylim_multiplier = 1.1

    def fetch_range(self, it_min=0, it_max=None):
        """
        Retrieves log data from the database.
        :param it_min: minimum iteration (inclusive)
        :param it_max: maximum iteration (inclusive)
        :return:
        """
        sort_by = "Iteration"
        con = None
        try:
            con, cur = self._get_db_con()
            if it_max:
                cur.execute(
                    "SELECT * FROM {} WHERE Phase='TRAIN' AND Iteration BETWEEN {} AND {} ORDER BY {} ASC".format(
                        self.log_db_prefix, it_min, it_max, sort_by))
            else:
                cur.execute("SELECT * FROM {} WHERE Phase='TRAIN' AND Iteration >= {} ORDER BY {} ASC".format(
                    self.log_db_prefix,
                    it_min, sort_by))
            self.cache_log_train = cur.fetchall()

            if it_max:
                cur.execute(
                    "SELECT * FROM {} WHERE Phase='TEST' AND Iteration BETWEEN {} AND {} ORDER BY {} ASC".format(
                        self.log_db_prefix, it_min, it_max, sort_by))
            else:
                cur.execute(
                    "SELECT * FROM {} WHERE Phase='TEST' AND Iteration >= {} ORDER BY {} ASC".format(self.log_db_prefix,
                                                                                                     it_min, sort_by))
            self.cache_log_test = cur.fetchall()

        except lite.Error, e:
            raise Exception('Error %s:' % e.args[0])
            # sys.exit(1)

        finally:
            if con:
                con.close()

        self._get_cached_iterations()
        self._unpack_cache()

    def fetch_last(self, samples=1):
        """
        Retrieves the most recent entries from the database
        :param samples: how many samples should be retrieved (will be used for both train and test logs).
        :return:
        """
        sort_by = "Iteration"
        con = None
        try:
            con, cur = self._get_db_con()
            cur.execute("SELECT * FROM {} WHERE Phase='TRAIN' ORDER BY {} DESC".format(self.log_db_prefix, sort_by))
            self.cache_log_train = cur.fetchmany(samples)

            cur.execute("SELECT * FROM {} WHERE Phase='TEST' ORDER BY {} DESC".format(self.log_db_prefix, sort_by))
            self.cache_log_test = cur.fetchmany(samples)

        except lite.Error, e:
            raise Exception('Error %s:' % e.args[0])
            # sys.exit(1)

        finally:
            if con:
                con.close()

        self._get_cached_iterations()
        self._unpack_cache()

    def _get_db_con(self, viz=False):
        suffix = "_viz" if viz else ""
        try:
            con = lite.connect(os.path.join(self.location, "{}{}.db".format(self.log_db_prefix, suffix)))
            return con, con.cursor()

        except lite.Error, e:
            raise Exception('Error %s:' % e.args[0])

    def _get_cached_iterations(self):
        self.cached_iterations = np.zeros(len(self.cache_log_train))
        for i, x in enumerate(self.cache_log_train):
            self.cached_iterations[i] = x[0]

    def _unpack_cache(self, viz=False):
        tmp = []
        idx_to_unpack = [0, 0, 1, 1, 1, 0, 0] if not viz else [0, 0, 1]
        for c in range(len(self.cache_log_train)):
            tmp.append(np.array(
                [json.loads(str(self.cache_log_train[c][idx])) if u else self.cache_log_train[c][idx] for idx, u in
                 enumerate(idx_to_unpack)]))
        self.cache_log_train = np.asarray(tmp)

        tmp = []
        for c in range(len(self.cache_log_test)):
            tmp.append(np.array(
                [json.loads(str(self.cache_log_test[c][idx])) if u else self.cache_log_test[c][idx] for idx, u in
                 enumerate(idx_to_unpack)]))
        self.cache_log_test = np.asarray(tmp)

        self.train_outputs = self.cache_log_train[0][2].keys()
        self.test_outputs = self.cache_log_test[0][2].keys()
        self.blob_percentiles_keys = self.cache_log_train[0][3].keys()
        self.weight_percentiles_keys = self.cache_log_train[0][4].keys()

    def plot_lr(self):
        """
        Plot the learning rate against iterations. You'll need to fetch data first.
        :return:
        """
        if not self.train_outputs and not self.test_outputs:
            print("You need to fetch data first using fetch_range() or fetch_last().")
            return

        f = plt.figure()
        plt.plot(self.cache_log_train[:, 0], self.cache_log_train[:, 1], label=r'lr')
        plt.xlabel('iteration')
        plt.legend(loc='best')
        plt.grid()

    def plot(self):
        """
        Plot the net's outputs against iterations. You'll need to fetch data first.
        :return:
        """
        if not self.train_outputs and not self.test_outputs:
            print("You need to fetch data first using fetch_range() or fetch_last().")
            return
        num_graphs = len(self.train_outputs + self.test_outputs)
        colormap = cm.winter(np.linspace(0, 1, num_graphs))
        graph_count = 0
        f = plt.figure()
        iterations = self.cache_log_train[:, 0].astype(int)
        for train_output in self.train_outputs:
            plt.plot(iterations, [x[train_output] for x in self.cache_log_train[:, 2]],
                     label=train_output, color=colormap[graph_count])
            graph_count += 1

        iterations = self.cache_log_test[:, 0].astype(int)
        for test_output in self.test_outputs:
            plt.plot(iterations, [x[test_output] for x in self.cache_log_test[:, 2]], label=test_output,
                     color=colormap[graph_count])
            graph_count += 1
        plt.xlabel('iteration')
        plt.legend(loc='best')
        plt.grid()

    def plot_percentiles(self):
        if not self.train_outputs and not self.test_outputs:
            print("You need to fetch data first using fetch_range() or fetch_last().")
            return

        num_rows = len(self.blob_percentiles_keys) + len(self.weight_percentiles_keys)

        colormap = cm.brg(np.linspace(0.5, 0.8, self.num_percentiles))
        colormap = [colormap[4], colormap[2], colormap[0], colormap[2], colormap[4]]

        iterations = self.cache_log_train[:, 0].astype(int)

        fig, axes = plt.subplots(num_rows, 2)
        fig.set_size_inches(15, 2*num_rows, forward=True)
        fig.tight_layout()
        row = 0

        for layer in self.blob_percentiles_keys:
            blob_percentiles_data = np.array([x[layer][0] for x in self.cache_log_train[:, 3]])
            blob_percentiles_diff = np.array([x[layer][1] for x in self.cache_log_train[:, 3]])
            self.make_subplot(axes, row, 0, layer, iterations, blob_percentiles_data, colormap)
            self.make_subplot(axes, row, 1, layer, iterations, blob_percentiles_diff, colormap)
            row += 1

            if layer in self.weight_percentiles_keys:
                weight_percentiles_data = np.array([x[layer][0] for x in self.cache_log_train[:, 4]])
                weight_percentiles_diff = np.array([x[layer][1] for x in self.cache_log_train[:, 4]])

                self.make_subplot(axes, row, 0, layer, iterations, weight_percentiles_data, colormap, 'weights')
                self.make_subplot(axes, row, 1, layer, iterations, weight_percentiles_diff, colormap, 'weights')
                row += 1

    def make_subplot(self, axes, row, col, layer, iterations, blob_percentiles_data, colormap, suffix='out'):
        for percentile in reversed(range(self.num_percentiles)):
            axes[row, col].plot(iterations, blob_percentiles_data[:, percentile],
                                label=self.percentile_labels[percentile], color=colormap[percentile])
            axes[row, col].grid()
            axes[row, col].fill_between(iterations, blob_percentiles_data[:, 0], blob_percentiles_data[:, 4],
                                        color='0.75',
                                        alpha=0.1)
            axes[row, col].fill_between(iterations, blob_percentiles_data[:, 1], blob_percentiles_data[:, 3],
                                        color='0.75',
                                        alpha=0.3)
            descr = 'data' if col == 0 else 'diff'
            axes[row, col].set_title(layer + '-' + suffix + '-' + descr, fontsize=10)
            axes[row, col].locator_params(tight=True, axis='y', nbins=5)
            axes[row, col].set_ylim([x*self.ylim_multiplier for x in axes[row, col].get_ylim()])
            if 0 < row < len(self.blob_percentiles_keys) + len(self.weight_percentiles_keys) - 1:
                axes[row, col].set_xticklabels([])
