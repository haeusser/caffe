#!/usr/bin/python

import os
from pymill.CNN.Definition import *
from math import ceil

#TODO
def standardTest(DeployBlock, generateNet=True):
    def Block(net, datasetName, output, prefix=None, use_augmentation_mean=True):
        blobs = net.namedBlobs()

        dataset = Dataset.get(name=datasetName, phase='TEST')

        img0, img1, flow_gt = dataset.flowLayer(net)

        from deploy import Block as Deploy
        flow_pred = Deploy(net, img0, img1, flow_gt, dataset.width(), dataset.height(), dataset.meanColors(), use_augmentation_mean)

        if output:
            out_path = 'output_%s_%s' % (prefix, datasetName) if prefix else 'output_%s' % datasetName
            os.system('mkdir -p %s' % out_path)

            f = open('%s/viewer.cfg' % out_path, 'w')
            f.write('2 2\n')
            f.write('0 0 -img0.ppm\n')
            f.write('1 0 -img1.ppm\n')
            f.write('0 1 -flow.flo\n')
            f.write('1 1 -gt.flo\n')
            f.close()

            net.writeImage(img0, folder=out_path, prefix='', suffix='-img0')
            net.writeImage(img1, folder=out_path, prefix='', suffix='-img1')
            net.writeFlow(flow_pred, folder=out_path, prefix='', suffix='-flow')
            net.writeFlow(flow_gt, folder=out_path, prefix='', suffix='-gt')

    if generateNet:
        net = Network()

        dataset = str(param('dataset'))
        if dataset is None:
            raise Exception('please specify dataset=...')

        use_augmentation_mean = bool(param('use_augmentation_mean', default=True))
        output = bool(param('output', default=False))
        prefix = str(param('prefix', default=None))

        Block(net,
              dataset,
              output,
              prefix,
              use_augmentation_mean)

        print net.toProto()

    return Block

def standardExtract(generateNet=True):
    def Block(net, datasetName, out_path='data'):
        blobs = net.namedBlobs()

        dataset = Dataset.get(name=datasetName, phase='TEST')

        img0, img1, flow_gt = dataset.flowLayer(net)

        os.system('mkdir -p %s' % out_path)

        f = open('%s/viewer.cfg' % out_path, 'w')
        f.write('3 2\n')
        f.write('0 0 -img0.ppm\n')
        f.write('1 0 -img1.ppm\n')
        f.write('2 1 -gt.flo\n')
        f.write('0 1 -ldof.flo\n')
        f.write('1 1 -df.flo\n')
        f.write('2 1 -ef.flo\n')
        f.close()

        net.writeImage(img0, folder=out_path, prefix='', suffix='-img0')
        net.writeImage(img1, folder=out_path, prefix='', suffix='-img1')
        net.writeFlow(flow_gt, folder=out_path, prefix='', suffix='-gt')

    if generateNet:
        net = Network()

        dataset = str(param('dataset'))
        if dataset is None:
            raise Exception('please specify dataset=...')

        Block(net, dataset)

        print net.toProto()

    return Block

def standardDeploy(NetworkBlock, generateNet=True):
    def Block(net, input_blobs, width, height, mean_color, augmentation_mean=True):
        blobs = net.namedBlobs()
        
        for key, value in input_blobs.iteritems():
          setattr(blobs, key, value)
          
        data_blobs = ('img0L'. 'img0R', 'img1L', 'img1R')
        gt_blobs = ('flowL', 'flowR', 'disp0L', 'disp1L', 'dispChangeL')

        for blob_name in data_blobs
          setattr(blobs, blob_name + 's', net.imageToRange01(getattr(blobs, blob_name)))

        divisor = 64.
        temp_width = ceil(width/divisor) * divisor
        temp_height = ceil(height/divisor) * divisor
        rescale_coeff_x = width / temp_width
        rescale_coeff_y = height / temp_height

        if augmentation_mean:
          for blob_name in data_blobs:
            setattr(blobs, blob_name + '_nomean', net.subtractAugmentationMean(getattr(blobs, blob_name + 's'), name=blob_name + "s_aug", width=width, height=height))
        else:
          for blob_name in data_blobs:
            setattr(blobs, blob_name + '_nomean', net.subtractMean(getattr(blobs, blob_name + 's'), mean_color))

        for blob_name in data_blobs:
          setattr(blobs, blob_name + '_nomean_resize', net.resample(getattr(blobs, blob_name + '_nomean'), width=temp_width, height=temp_height, type='LINEAR', antialias=True))

        for blob_name in gt_blobs:
         setattr(blobs, blob_name + '_gt_resize', net.resample(blob_name, width=temp_width, height=temp_height, type='NEAREST', antialias=True) # Use NEAREST here, since kitti gt is sparse
                 
        net_blobs = {}
        for blob_name in data_blobs:
          net_blobs[blob_name + '_nomean_resize'] = getattr(blobs, blob_name + '_nomean_resize')
        for blob_name in gt_blobs:
          net_blobs[blob_name + '_gt_resize'] = getattr(blobs, blob_name + '_gt_resize')

        from net import Block as Network
        prediction = NetworkBlock(net, net_blobs)

        for blob_name in gt_blobs:
          setattr(blobs, 'predict_' + blob_name + '_resize', net.resample(prediction[blob_name], width=width, height=height, reference=None, type='LINEAR', antialias=True)
          setattr(blobs, 'predict_' + blob_name + '_final', net.scale(getattr(blobs, 'predict_' + blob_name + '_resize'), (rescale_coeff_x, rescale_coeff_y))

        epe_losses = {}
        for blob_name in gt_blobs:
          epe_losses[blob_name] = Layers.L1Loss(net, (getattr(blobs,blob_name), getattr(blobs,'predict_' + blob_name + '_final')), nout=1, loss_weight=(1,), name='EPE_' + blob_name, l2_per_location=True, normalize_by_num_entries=True, epsilon=0)
          #TODO epe_loss.setName('epe')
          #TODO epe_loss.enableOutput()

        return blobs.predict_flow_final

    if generateNet:
        net = Network()
        Block(net)
        print net.toProto()

    return Block
