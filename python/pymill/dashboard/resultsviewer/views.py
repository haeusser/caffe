from django.shortcuts import render
from django_tables2 import RequestConfig
from django.db.models import Sum
from resultsviewer.models import Results
from resultsviewer.tables import ResultsTable
from resultsviewer.tables import FormattedResults
from resultsviewer.tables import FormattedResultsTable
from .forms import OptionsForm
import json


def results(request):
    queryset = Results.objects.using('results')

    measures = [x.values() for x in queryset.all().values('measure').distinct()]
    positions = [x.values() for x in queryset.all().values('position').distinct()]
    datasets = [x.values() for x in queryset.all().values('dataset').distinct()]
    nets = [x.values() for x in queryset.all().values('networkname').distinct()]

    filter_params = dict()

    cookie_set = True if 'filter_params' in request.COOKIES else False  # TODO check age of cookie!
    new_cookie_necessary = False

    if cookie_set and 'order_by' in json.loads(request.COOKIES.get('filter_params')):
        filter_params['order_by'] = json.loads(request.COOKIES.get('filter_params'))['order_by']

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
            if not filter_params['selected_measure'] == '':
                queryset = queryset.filter(measure=selected_measure)
                # table.exclude += ('measure',)

        if 'selected_position' in filter_params:
            selected_position = filter_params['selected_position']
            if not filter_params['selected_position'] == '':
                queryset = queryset.filter(position=selected_position)

        if 'only_last_iteration' in filter_params and filter_params['only_last_iteration'] == 'on':
            ids = []
            for net in nets:
                for ds in datasets:
                    for m in measures:
                        q = queryset.filter(networkname=net[0]).filter(dataset=ds[0]).filter(measure=m[0]).order_by('-iteration').values()
                        if len(q) > 0:
                            ids.append(q[0]['id'])

            queryset = queryset.filter(id__in=ids)

    if request.method == 'GET':
        selected_measure = 'none'
        if 'sort' in request.GET:
            filter_params['order_by'] = request.GET['sort']
            new_cookie_necessary = True

    try:  # possibly, the queryset is empty
        best = queryset.order_by('value')[0]
        best_net = best.networkname
        best_dataset = best.dataset
        best_value = best.value
        best_measure = best.measure
    except:
        pass

    sorting = filter_params['order_by'] if 'order_by' in filter_params else 'networkname'

    formatted_queryset = reorganize_queryset(queryset)
    table = FormattedResultsTable(formatted_queryset, order_by=sorting)

    for c in table.columns.items():
        cname = c[0]
        if cname.endswith(tuple(['clean', 'final', 'train', 'val', 'test'])):
            col = table.columns[cname]
            col.attrs['td'].update({'align': 'right'})

    RequestConfig(request, paginate=False).configure(table)

    table.exclude += ('id',)
    table.exclude += ('dataset', 'value',)

    filter_form = OptionsForm(filter_params, measures=measures, positions=positions)
    response = render(request, 'results.html', locals())

    if new_cookie_necessary:
        response.set_cookie(key='filter_params', value=json.dumps(filter_params))
    return response


def reorganize_queryset(queryset):
    q = queryset.all()
    nets = q.values_list('networkname', flat=True)
    nets_set = set(nets)
    records = []
    datasets = [
                'sintel.train.clean',
                'sintel.train.final',
                'monkaa.test.clean',
                'monkaa.test.final',
                'FlyingStuff3D.test.clean',
                'FlyingStuff3D.test.final',
                'kitti2012.train',
                'kitti2015.train',
                'chairs.val'
                ]

    for net in nets_set:
        iterations = set(q.filter(networkname=net).values_list('iteration', flat=True))
        measures = set(q.filter(networkname=net).values_list('measure', flat=True))
        positions = set(q.filter(networkname=net).values_list('position', flat=True))

        for i in iterations:
            for m in measures:
                for p in positions:

                    record = {
                        'networkname': net,
                        'iteration': i,
                        'measure': m,
                        'position': p
                    }

                    for d in datasets:
                        result_sum = q.filter(networkname=net, dataset=d, iteration=i, measure=m, position=p).aggregate(dataset_sum=Sum('value'))
                        record[d] = result_sum['dataset_sum']

                    records.append(record)

    for record in records:
        for key in record:
            if '.' in key:
                value = record[key]
                #record[key] = format(float(record[key]), '.2f') if value else None
                record[key.replace('.', '')] = record.pop(key)

    return records


##### NOT IN USE CURRENTLY ####
"""
def create_model(name, fields=None, app_label='', module='', options=None, parent=models.Model):

    from django.db import models
    class Meta:
        # Using type('Meta', ...) gives a dictproxy error during model creation
        pass

    if app_label:
        # app_label must be set using the Meta inner class
        setattr(Meta, 'app_label', app_label)

    # Update Meta with any options that were provided
    if options is not None:
        for key, value in options.iteritems():
            setattr(Meta, key, value)

    # Set up a dictionary to simulate declarations within a class
    attrs = {'__module__': module, 'Meta': Meta}

    # Add in any fields that were provided
    if fields:
        attrs.update(fields)

    # Create the class, which automatically triggers ModelBase processing
    model = type(name, (parent,), attrs)

    return model


def make_formatted_model(datasets):
    from resultsviewer.models import Results
    from django.db import models

    attrs = dict()

    for d in datasets:
        attrs[d[0]] = models.TextField(blank=True, null=True)

    return create_model('FormattedResults', fields=attrs, app_label='resultsviewer', parent=Results)


def make_formatted_table(model):
    import django_tables2 as tables
    options = {
         = FormattedResults
        attrs = {"class": "paleblue"}
        'model': ['last_name', 'first_name'],
        'verbose_name': 'valued customer',
    }
    return create_model('FormattedResultsTable', app_label='resultsviewer', parent=tables.Table)
"""