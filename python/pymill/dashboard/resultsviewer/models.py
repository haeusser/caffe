from django.db import models


class Results(models.Model):
    networkname = models.TextField(db_column='networkName', blank=True, null=True, verbose_name="Network Name")  # Field name made lowercase.
    iteration = models.IntegerField(verbose_name="Iteration")
    dataset = models.TextField(blank=True, null=True, verbose_name="Dataset")
    measure = models.TextField(blank=True, null=True, verbose_name="Error Measure")
    position = models.TextField(blank=True, null=True, verbose_name="Position")
    value = models.FloatField(verbose_name="Error Value")

    class Meta:
        managed = False
        db_table = 'results'

    def save(self, *args, **kwargs):
        pass
        #raise NotImplemented


class FormattedResults(models.Model):
    networkname = models.TextField(db_column='networkName', blank=True, null=True, verbose_name="Network Name")  # Field name made lowercase.
    iteration = models.IntegerField(verbose_name="Iteration")
    measure = models.TextField(blank=True, null=True, verbose_name="Error Measure")
    position = models.TextField(blank=True, null=True, verbose_name="Position")

    # TODO: get these from CNN.Definition.Dataset.getDatasetNames()
    sinteltrainclean = models.FloatField(blank=True, null=True, verbose_name='sintel.train.clean', db_column='sintel.train.clean')
    sinteltrainfinal = models.FloatField(blank=True, null=True, verbose_name='sintel.train.final', db_column='sintel.train.final')
    monkaatestclean = models.FloatField(blank=True, null=True, verbose_name='monkaa.test.clean', db_column='monkaa.test.clean')
    monkaatestfinal = models.FloatField(blank=True, null=True, verbose_name='monkaa.test.final', db_column='monkaa.test.final')
    FlyingStuff3Dtestclean = models.FloatField(blank=True, null=True, verbose_name='FlyingStuff3D.test.clean', db_column='FlyingStuff3D.test.clean')
    FlyingStuff3Dtestfinal = models.FloatField(blank=True, null=True, verbose_name='FlyingStuff3D.test.final', db_column='FlyingStuff3D.test.final')
    FlyingStuff3Dnewtestclean = models.FloatField(blank=True, null=True, verbose_name='FlyingStuff3D_new.test.clean', db_column='FlyingStuff3D_new.test.clean')
    FlyingStuff3Dnewtestfinal = models.FloatField(blank=True, null=True, verbose_name='FlyingStuff3D_new.test.final', db_column='FlyingStuff3D_new.test.final')
    kitti2012train = models.FloatField(blank=True, null=True, verbose_name='kitti2012.train', db_column='kitti2012.train')
    kitti2015train = models.FloatField(blank=True, null=True, verbose_name='kitti2015.train', db_column='kitti2015.train')
    chairsval = models.FloatField(blank=True, null=True, verbose_name='chairs.val', db_column='chairs.val')

    class Meta:
        managed = False
        db_table = 'formattedresults'

    def save(self, *args, **kwargs):
        pass
        #raise NotImplemented