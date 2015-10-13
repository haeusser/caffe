import django_tables2 as tables
from resultsviewer.models import Results

class ResultsTable(tables.Table):
    class Meta:
        model = Results
        attrs = {"class": "paleblue"}
        #fields = ('networkname', 'iteration', 'dataset', 'measure', 'value') #, 'position', )


