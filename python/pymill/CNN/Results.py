#!/usr/bin/python

import os
import sqlite3 as lite

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
        cur.execute('SELECT value FROM results WHERE networkName ="%s" AND iteration="%d" AND dataset="%s" AND measure="%s"' % (self._net, int(iter), dataset, measure))
        return cur.fetchone()[0]

    def update(self, iter, dataset, measure, value):
        self._conn.execute('UPDATE results SET value = %f WHERE networkName ="%s" AND iteration="%d" AND dataset="%s" AND measure="%s"' % (float(value), self._net, int(iter), dataset, measure))
        if self._conn.total_changes == 0:
            self._conn.execute('INSERT INTO results VALUES ("%s", %d, "%s", "%s", %f)' % (self._net, iter, dataset, measure, float(value)))
        self._conn.commit()

