from django.shortcuts import render
from django_tables2 import RequestConfig
from resultsviewer.models import Results
from resultsviewer.tables import ResultsTable
from .forms import OptionsForm
import json


def results(request):
    queryset = Results.objects.using('results')

    measures = [x.values() for x in queryset.all().values('measure').distinct()]

    filter_params = dict()
    if 'order_by' in json.loads(request.COOKIES.get('filter_params')):
        filter_params['order_by'] = json.loads(request.COOKIES.get('filter_params'))['order_by']

    cookie_set = True if 'filter_params' in request.COOKIES else False  # TODO check age of cookie!
    new_cookie_necessary = False

    if request.method == 'POST' or cookie_set:
        if request.method == 'POST':
            filter_params.update(request.POST)
            new_cookie_necessary = True
        else:
            filter_params.update(json.loads(request.COOKIES.get('filter_params')))
        for key in filter_params:
            value = filter_params[key]
            filter_params[key] = value[0] if type(value) == list and len(value) == 1 else value

        if 'selected_measure' in filter_params:
            selected_measure = filter_params['selected_measure']
            measure_selector_form = OptionsForm(filter_params, measures=measures)
            if not filter_params['selected_measure'] == 'all':
                queryset = queryset.filter(measure=selected_measure)
                # table.exclude += ('measure',)

        if 'only_last_iteration' in filter_params and filter_params['only_last_iteration'] == 'on':
            ids = []
            nets = [x.values() for x in queryset.all().values('networkname').distinct()]
            for net in nets:
                ids.append(queryset.filter(networkname=net[0]).order_by('-iteration').values()[0]['id'])
            queryset = queryset.filter(id__in=ids)

    if request.method == 'GET':
        measure_selector_form = OptionsForm(filter_params, measures=measures)
        selected_measure = 'none'
        if 'sort' in request.GET:
            filter_params['order_by'] = request.GET['sort']
            new_cookie_necessary = True

    sorting = filter_params['order_by'] if 'order_by' in filter_params else 'value'
    table = ResultsTable(queryset.all(), order_by=sorting)
    RequestConfig(request, paginate=False).configure(table)
    table.exclude += ('id',)

    best = queryset.order_by('value')[0]
    best_net = best.networkname
    best_dataset = best.dataset
    best_value = best.value
    best_measure = best.measure

    response = render(request, 'results.html', locals())

    if new_cookie_necessary:
        response.set_cookie(key='filter_params', value=json.dumps(filter_params))

    return response
