from django.shortcuts import render
from django_tables2   import RequestConfig
from resultsviewer.models  import Results
from resultsviewer.tables  import ResultsTable
from .forms import OptionsForm
def results(request):
    queryset = Results.objects.using('results')

    measures = [x.values() for x in queryset.all().values('measure').distinct()]

    if request.method == 'GET':
        measure_selector_form = OptionsForm(measures=measures)
        selected_measure = 'none'

    if request.method == 'POST':
        selected_options = request.POST
        if 'selected_measure' in request.POST:
            selected_measure = request.POST['selected_measure']
            measure_selector_form = OptionsForm(request.POST, measures=measures)
            if not request.POST['selected_measure'] == 'all':
                queryset = queryset.filter(measure=selected_measure)
                #table.exclude += ('measure',)
        if 'only_last_iteration' in request.POST and request.POST['only_last_iteration'] == 'on':
            selected_options = "ONLY_LAST!"
            ids = []
            nets = [x.values() for x in queryset.all().values('networkname').distinct()]
            for net in nets:
                ids.append(queryset.filter(networkname=net[0]).order_by('-iteration').values()[0]['id'])
            queryset = queryset.filter(id__in=ids)

    table = ResultsTable(queryset.all(), order_by='value')
    RequestConfig(request, paginate=False).configure(table)
    table.exclude += ('id',)
    #table.order_by = 'value'

    best = queryset.order_by('value')[0]
    best_net = best.networkname
    best_dataset = best.dataset
    best_value = best.value
    best_measure = best.measure

    return render(request, 'results.html', locals()) # {'table': table})
