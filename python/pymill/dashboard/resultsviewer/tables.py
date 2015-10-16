import django_tables2 as tables
from resultsviewer.models import Results
from resultsviewer.models import FormattedResults


class ResultsTable(tables.Table):
    class Meta:
        model = Results
        attrs = {"class": "paleblue"}


class FormattedResultsTable(tables.Table):
    class Meta:
        model = FormattedResults
        attrs = {"class": "paleblue"}