#!/usr/bin/python

import os
from pymill.CNN.Definition import *
from math import ceil

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
    def Block(net, datasetName, output_path='data'):
        blobs = net.namedBlobs()

        dataset = Dataset.get(name=datasetName, phase='TEST')

        img0, img1, flow_gt = dataset.flowLayer(net)

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

def standardDeploy(NetworkBlock, generateNet=True):
    def Block(net, img0, img1, flow_gt, width, height, mean_color, augmentation_mean=True):
        blobs = net.namedBlobs()

        blobs.img0 = img0
        blobs.img1 = img1
        blobs.flow_gt = flow_gt

        blobs.img0s = net.imageToRange01(blobs.img0)
        blobs.img1s = net.imageToRange01(blobs.img1)

        divisor = 64.
        temp_width = ceil(width/divisor) * divisor
        temp_height = ceil(height/divisor) * divisor
        rescale_coeff_x = width / temp_width
        rescale_coeff_y = height / temp_height

        if augmentation_mean:
            blobs.img0_nomean = net.subtractAugmentationMean(blobs.img0s, name="img0s_aug", width=width, height=height)
            blobs.img1_nomean = net.subtractAugmentationMean(blobs.img1s, name="img1s_aug", width=width, height=height)
        else:
            blobs.img0_nomean = net.subtractMean(blobs.img0s, mean_color)
            blobs.img1_nomean = net.subtractMean(blobs.img1s, mean_color)

        blobs.img0_nomean_resize = net.resample(blobs.img0_nomean, width=temp_width, height=temp_height, type='LINEAR', antialias=True)
        blobs.img1_nomean_resize = net.resample(blobs.img1_nomean, width=temp_width, height=temp_height, type='LINEAR', antialias=True)
        blobs.flow_gt_resize     = net.resample(blobs.flow_gt,     width=temp_width, height=temp_height, type='NEAREST', antialias=True) # Use NEAREST here, since kitti gt is sparse

        from net import Block as Network
        prediction = NetworkBlock(net,
                             blobs.img0_nomean_resize,
                             blobs.img1_nomean_resize,
                             blobs.flow_gt_resize)

        blobs.predict_flow_resize = net.resample(prediction, width=width, height=height, reference=None, type='LINEAR', antialias=True)
        blobs.predict_flow_final  = net.scale(blobs.predict_flow_resize, (rescale_coeff_x, rescale_coeff_y))

        epe_loss = Layers.L1Loss(net, (blobs.flow_gt, blobs.predict_flow_final), nout=1, loss_weight=(1,), name='EPE', l2_per_location=False, normalize_by_num_entries=True, epsilon=0)
        epe_loss.setName('epe')
        epe_loss.enableOutput()

        return blobs.predict_flow_final

    if generateNet:
        net = Network()
        Block(net)
        print net.toProto()

    return Block
