#!/usr/bin/python

import os
import sqlite3 as lite
import datetime

basePath = "/misc/lmbraid17/sceneflownet/common"
dbPath = "/misc/lmbraid17/sceneflownet/common/results/db.sqlite"

class Results:
    @staticmethod
    def readAll():
        conn = lite.connect(dbPath)
        conn.row_factory = lite.Row
        cur = conn.cursor(lite.Cursor)
        cur.execute('SELECT * FROM results')
        return cur.fetchall()

    def __init__(self, path):
        self._conn = lite.connect(dbPath)
        self._net = path.replace(basePath+'/nets/','')

    def read(self, iter, dataset, measure):
        cur = self._conn.cursor()
        dataset = dataset.replace('_', '.')
        cur.execute('SELECT value FROM results WHERE networkName ="%s" AND iteration="%d" AND dataset="%s" AND measure="%s"' % (self._net, int(iter), dataset, measure))
        return cur.fetchone()[0]

    def update(self, iter, dataset, task, measure, position, value):
        if position is None: position = ""
        dataset = dataset.replace('_', '.')
        self._conn.execute('UPDATE results SET value = %f, date = "%s" WHERE networkName ="%s" AND iteration="%d" AND dataset="%s" AND task="%s" AND measure="%s" AND position="%s"' % (float(value), str(datetime.datetime.now()), self._net, int(iter), dataset, task, measure, position))
        if self._conn.total_changes == 0:
            self._conn.execute('INSERT INTO results (networkName, task, iteration, dataset, measure, position, value, date) VALUES ("%s", "%s", %d, "%s", "%s", "%s", %f, "%s")' % (self._net, task, iter, dataset, measure, position, float(value), str(datetime.datetime.now())))
        self._conn.commit()

