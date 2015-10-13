from django.shortcuts import render
from django_tables2   import RequestConfig
from resultsviewer.models  import Results
from resultsviewer.tables  import ResultsTable
from .forms import MeasureSelectorForm
def results(request):
    queryset = Results.objects.using('results')
    table = ResultsTable(queryset.all())


    best = queryset.order_by('value')[0]
    best_net = best.networkname
    best_dataset = best.dataset
    best_value = best.value
    best_measure = best.measure

    measures = [x.values() for x in queryset.all().values('measure').distinct()]

    RequestConfig(request, paginate=False).configure(table)

    if request.method == 'GET':
        measure_selector_form = MeasureSelectorForm()
        selected_measure = 'none'

    if request.method == 'POST':
        if 'selected_measure' in request.POST:
            selected_measure = request.POST['selected_measure']
            measure_selector_form = MeasureSelectorForm(request.POST)
            if not request.POST['selected_measure'] == 'all':
                table = ResultsTable(queryset.filter(measure=selected_measure).all())
                #table.exclude += ('measure',)
            else:
                table = ResultsTable(queryset.all())


    table.exclude += ('id',)
    table.order_by = 'value'

    return render(request, 'results.html', locals()) # {'table': table})
