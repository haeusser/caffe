from django.db import models

"""
class Result(models.Model):
    net_name = models.CharField(verbose_name='network name', max_length=100)
    iteration = models.IntegerField(verbose_name='iteration')
    dataset = models.CharField(verbose_name='dataset', max_length=100)
    measure = models.CharField(verbose_name='error measure', max_length=100)
    value = models.FloatField(verbose_name='error value')
"""

class Results(models.Model):
    networkname = models.TextField(db_column='networkName', blank=True, null=True, verbose_name="Network Name")  # Field name made lowercase.
    iteration = models.IntegerField(verbose_name="Iteration")
    dataset = models.TextField(blank=True, null=True, verbose_name="Dataset")
    measure = models.TextField(blank=True, null=True, verbose_name="Error Measure")
    value = models.FloatField(verbose_name="Error Value")

    class Meta:
        managed = False
        db_table = 'results'

    def save(self, *args, **kwargs):
        pass
        #raise NotImplemented