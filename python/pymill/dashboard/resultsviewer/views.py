from django.shortcuts import render
from django_tables2   import RequestConfig
from resultsviewer.models  import Results
from resultsviewer.tables  import ResultsTable
import sqlite3 as lite

"""
def people(request):
    table = PersonTable(Person.objects.all())
    RequestConfig(request).configure(table)
    return render(request, 'people.html', {'table': table})
"""

def results(request):
    queryset = Results.objects.using('results')
    table = ResultsTable(queryset.all())
    best = queryset.order_by('value')[0]
    best_net = best.networkname
    best_dataset = best.dataset
    best_value = best.value
    best_measure = best.measure
    RequestConfig(request).configure(table)
    """
    dbPath = '/home/haeusser/libs/hackathon-caffe2/python/pymill/test/results/db.sqlite'
    conn = lite.connect(dbPath)
    query = ''  # TODO set to live DB
    try:
        conn.row_factory = lite.Row
        cur = conn.cursor(lite.Cursor)
        cur.execute('SELECT * FROM results')
        query = cur.fetchall()
    finally:
        conn.close()
    """


    return render(request, 'results.html', locals()) # {'table': table})
