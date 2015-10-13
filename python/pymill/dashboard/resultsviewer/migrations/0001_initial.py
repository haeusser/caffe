# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Results',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('networkname', models.TextField(null=True, verbose_name=b'Network Name', db_column=b'networkName', blank=True)),
                ('iteration', models.IntegerField(verbose_name=b'Iteration')),
                ('dataset', models.TextField(null=True, verbose_name=b'Dataset', blank=True)),
                ('measure', models.TextField(null=True, verbose_name=b'Error Measure', blank=True)),
                ('value', models.FloatField(verbose_name=b'Error Value')),
            ],
            options={
                'db_table': 'results',
                'managed': False,
            },
        ),
    ]
