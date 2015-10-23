#!/usr/bin/python
# -*- coding: utf-8 -*-

##
# Scene flow
##

import os
from pymill.CNN.Definition import *
from math import ceil

def standardTest(DeployBlock, generateNet=True):
    '''
    @brief Create Testing-mode wrapper for a Deploy-mode net
    
    @param DeployBlock Deploy-mode Block(..) function
    @param generateNet IFF TRUE, a network prototxt will be made and printed
    
    @returns Testing-mode Block(..) function that adds a data layer to a Deploy-mode
             net
    '''
    def Block(net, datasetName, output, prefix=None, use_augmentation_mean=True, have_flowR=True):
        '''
        @brief Add a data layer for dataset "datasetName" to a network
        
        @param net Incomplete network definition
        @param datasetName Name of the desired dataset (see Dataset.py)
        @param output IFF TRUE, the network will write its input and output to disk
        @param prefix Filename prefix for file outputs of "output" is TRUE
        @param use_augmentation_mean IFF TRUE, data mean will be computed on the fly
        '''
        blobs = net.namedBlobs()

        ## Make data layer
        dataset = Dataset.get(name=datasetName, phase='TEST')
        data = dataset.sceneFlowLayer(net)

        data.pred = DeployBlock(net,
                                data,
                                dataset.width(),
                                dataset.height(),
                                dataset.meanColors(),
                                use_augmentation_mean)

        ## Output network input and output
        if output:
            if prefix:
                out_path = 'output_%s_%s' % (prefix, datasetName) 
            else:
                out_path = 'output_%s' % datasetName
            if not os.path.isdir(out_path):
                os.makedirs(out_path)

            ## Write configuration file for viewer tool
            #  ┌───────────┬───────────┬────────────┬─────────────┐
            #  │    L_t    │    R_t    │ disp0L_gt  │ disp0L_pred │
            #  ├───────────┼───────────┼────────────┼─────────────┤
            #  │   L_t+1   │   R_t+1   │ disp1L_gt  │ disp1L_pred │
            #  ├───────────┼───────────┼────────────┼─────────────┤
            #  │ flowL_gt  │ flowR_gt  │ dispChL_gt │     ---     │
            #  ├───────────┼───────────┼────────────┼─────────────┤
            #  │flowL_pred │flowR_pred │dispChL_pred│     ---     │
            #  └───────────┴───────────┴────────────┴─────────────┘
            ##
            f = open('%s/viewer.cfg' % out_path, 'w')
            f.write('4 4\n')
            f.write('0 0 -img0L.ppm\n')
            f.write('1 0 -img0R.ppm\n')
            f.write('2 0 -disp0L_gt.float3\n')
            f.write('3 0 -disp0L_pred.float3\n')
            f.write('0 1 -img1L.ppm\n')
            f.write('1 1 -img1R.ppm\n')
            f.write('2 1 -disp1L_gt.float3\n')
            f.write('3 1 -disp1L_pred.float3\n')
            f.write('0 2 -flowL_gt.flo\n')
            if have_flowR: f.write('1 2 -flowR_gt.flo\n')
            else:          f.write('1 2 none\n')
            f.write('2 2 -dispChangeL_gt.float3\n')
            f.write('3 2 none\n')
            f.write('0 3 -flowL_pred.flo\n')
            if have_flowR: f.write('1 3 -flowR_pred.flo\n')
            else:          f.write('1 3 none\n')
            f.write('2 3 -dispChangeL_pred.float3\n')
            f.write('3 3 none\n')
            f.close()

            net.writeImage(data.inp.img0L,        folder=out_path, prefix='', suffix='-img0L')
            net.writeImage(data.inp.img0R,        folder=out_path, prefix='', suffix='-img0R')
            net.writeImage(data.inp.img1L,        folder=out_path, prefix='', suffix='-img1L')
            net.writeImage(data.inp.img1R,        folder=out_path, prefix='', suffix='-img1R')
            net.writeFlow(data.gt.flowL,          folder=out_path, prefix='', suffix='-flowL_gt')
            if have_flowR:
                net.writeFlow(data.gt.flowR,          folder=out_path, prefix='', suffix='-flowR_gt')
            net.writeFloat(data.gt.disp0L,        folder=out_path, prefix='', suffix='-disp0L_gt')
            net.writeFloat(data.gt.disp1L,        folder=out_path, prefix='', suffix='-disp1L_gt')
            net.writeFloat(data.gt.dispChangeL,   folder=out_path, prefix='', suffix='-dispChangeL_gt')
            net.writeFlow(data.pred.flowL,        folder=out_path, prefix='', suffix='-flowL_pred')
            if have_flowR:
                net.writeFlow(data.pred.flowR,        folder=out_path, prefix='', suffix='-flowR_pred')
            net.writeFloat(data.pred.disp0L,      folder=out_path, prefix='', suffix='-disp0L_pred')
            net.writeFloat(data.pred.disp1L,      folder=out_path, prefix='', suffix='-disp1L_pred')
            net.writeFloat(data.pred.dispChangeL, folder=out_path, prefix='', suffix='-dispChangeL_pred')

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
    '''
    @brief Create a network wrapper that dumps dataset contents

    @param generateNet IFF TRUE, a network prototxt will be made and printed

    @returns Block(..) function that adds a data layer and dumping functions to a net
    '''
    def Block(net, datasetName, out_path='data'):
        '''
        @brief Add data a data layer and file outputs to a network
        
        @param net Incomplete network definition
        @param datasetName Name of the desired dataset (see Dataset.py)
        @param out_path Output path for file dumps
        '''
        blobs = net.namedBlobs()
        ## Create data layer
        dataset = Dataset.get(name=datasetName, phase='TEST')
        img0L, img0R, img1L, img1R,  \
          flowL_gt, flowR_gt,        \
          disp0L_gt, disp1L_gt, dispChangeL_gt = dataset.sceneFlowLayer(net)

        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        ## Write configuration file for viewer tool
        f = open('%s/viewer.cfg' % out_path, 'w')
        f.write('3 3\n')
        f.write('0 0 -img0L.ppm\n')
        f.write('1 0 -img0R.ppm\n')
        f.write('2 0 -disp0L_gt.floa3t\n')
        f.write('0 1 -img1L.ppm\n')
        f.write('1 1 -img1R.ppm\n')
        f.write('2 1 -disp1L_gt.float3\n')
        f.write('0 2 -flowL_gt.flo\n')
        f.write('1 2 -flowR_gt.flo\n')
        f.write('2 2 -dispChangeL_gt.float3\n')
        f.close()

        net.writeImage(img0L,     folder=out_path, prefix='', suffix='-img0L')
        net.writeImage(img0R,     folder=out_path, prefix='', suffix='-img0R')
        net.writeImage(img1L,     folder=out_path, prefix='', suffix='-img1L')
        net.writeImage(img1R,     folder=out_path, prefix='', suffix='-img1R')
        net.writeFlow(flowL_gt,   folder=out_path, prefix='', suffix='-flowL_gt')
        net.writeFlow(flowR_gt,   folder=out_path, prefix='', suffix='-flowR_gt')
        net.writeFloat(disp0L_gt, folder=out_path, prefix='', suffix='-disp0L_gt')
        net.writeFloat(disp1L_gt, folder=out_path, prefix='', suffix='-disp1L_gt')
        net.writeFloat(dispChangeL_gt, folder=out_path, prefix='', suffix='-dispChangeL_gt')

    if generateNet:
        net = Network()

        dataset = str(param('dataset'))
        if dataset is None:
            raise Exception('please specify dataset=...')

        Block(net, dataset)

        print net.toProto()

    return Block

def standardAugmentationTest(AugmentationBlock, generateNet=True):
    pass
    #unfinished code:
    # datasetName = str(param('dataset'))
    # if datasetName is None:
    #     raise Exception('please specify dataset=...')
    #
    # dataset = Dataset.get(name=datasetName, phase='TEST')
    # data = dataset.sceneFlowLayer(net)
    #
    # data.inpAug, data.gtAug = Block(net, data, batch_size=4, in_width=dataset.width(), in_height=dataset.height())
    #
    # os.system('mkdir -p output_augmentation_test')
    # f = open('%s/viewer.cfg' % out_path, 'w')
    # f.write('4 4\n')
    # f.write('0 0 -img0L.ppm\n')
    # f.write('1 0 -img0R.ppm\n')
    # f.write('2 0 -disp0L_gt.float3\n')
    # f.write('3 0 -disp0L_pred.float3\n')
    # f.write('0 1 -img1L.ppm\n')
    # f.write('1 1 -img1R.ppm\n')
    # f.write('2 1 -disp1L_gt.float3\n')
    # f.write('3 1 -disp1L_pred.float3\n')
    # f.write('0 2 -flowL_gt.flo\n')
    # f.write('1 2 -flowR_gt.flo\n')
    # f.write('2 2 -dispChangeL_gt.float3\n')
    # f.write('3 2 none\n')
    # f.write('0 3 -flowL_pred.flo\n')
    # f.write('1 3 -flowR_pred.flo\n')
    # f.write('2 3 -dispChangeL_pred.float3\n')
    # f.write('3 3 none\n')
    # f.close()
    #
    # net.writeImage(data.inp.img0L,        folder=out_path, prefix='', suffix='-img0L')
    # net.writeImage(data.inp.img0R,        folder=out_path, prefix='', suffix='-img0R')
    # net.writeImage(data.inp.img1L,        folder=out_path, prefix='', suffix='-img1L')
    # net.writeImage(data.inp.img1R,        folder=out_path, prefix='', suffix='-img1R')
    # net.writeFlow(data.gt.flowL,          folder=out_path, prefix='', suffix='-flowL_gt')
    # net.writeFlow(data.gt.flowR,          folder=out_path, prefix='', suffix='-flowR_gt')
    # net.writeFloat(data.gt.disp0L,        folder=out_path, prefix='', suffix='-disp0L_gt')
    # net.writeFloat(data.gt.disp1L,        folder=out_path, prefix='', suffix='-disp1L_gt')
    # net.writeFloat(data.gt.dispChangeL,   folder=out_path, prefix='', suffix='-dispChangeL_gt')
    # net.writeFlow(data.gtAug.flowL,        folder=out_path, prefix='', suffix='-flowL_pred')
    # net.writeFlow(data.pred.flowR,        folder=out_path, prefix='', suffix='-flowR_pred')
    # net.writeFloat(data.pred.disp0L,      folder=out_path, prefix='', suffix='-disp0L_pred')
    # net.writeFloat(data.pred.disp1L,      folder=out_path, prefix='', suffix='-disp1L_pred')
    # net.writeFloat(data.pred.dispChangeL, folder=out_path, prefix='', suffix='-dispChangeL_pred')


def standardDeploy(NetworkBlock, generateNet=True):
    '''
    @brief Create Deploy-mode wrapper for a raw net
    
    @param NetworkBlock Block(..) function of a raw network
    @param generateNet IFF TRUE, a network prototxt will be made and printed
    
    @returns Deploy-mode Block(..) function that adds data preprocessing and results postprocessing to a raw network
    '''
    def Block(net, data, width, height, mean_color, augmentation_mean=True):
        '''
        @brief Add data preprocessing to a network and connect data inputs
        
        @param net Incomplete network definition
        @param data DataStruct
        @param width Input width (pixels)
        @param height Input height (pixels)
        @param mean_color Data mean to be subtracted from data if "augmentation_mean" is FALSE
        @param augmentation_mean IFF TRUE, data mean will be computed on the fly
        
        @returns DataStruct with predictions
        '''
        ## Rescale input images to [0,1]
        for name in data.inp:
            data.scaled[name] = net.imageToRange01(data.inp[name])

        ## Subtract given mean or connect mean computation layer
        if augmentation_mean:
            for name in data.scaled:
                data.nomean[name] = \
                    net.subtractAugmentationMean(data.scaled[name],
                                                 name=name + 's_aug',
                                                 width=width,
                                                 height=height)
        else:
            for name in data.scaled:
                data.nomean[name] = net.subtractMean(data.scaled[name], mean_color)

        ## Resample input data (needs to be 64-pixels aligned)
        divisor = 64.
        temp_width = ceil(width/divisor) * divisor
        temp_height = ceil(height/divisor) * divisor
        rescale_coeff_x = width / temp_width
        rescale_coeff_y = height / temp_height

        for name in data.nomean:
            data.nomean_resized[name] = \
                net.resample(data.nomean[name],
                             width=temp_width,
                             height=temp_height,
                             type='LINEAR',
                             antialias=True)

        for name in data.gt:
            data.gt_resized[name] = net.resample(data.gt[name],
                                                 width=temp_width,
                                                 height=temp_height,
                                                 type='NEAREST',
                                                 antialias=True)

        ## Connect data preprocessing layers to raw net
        data.pred_raw = NetworkBlock(net, data.nomean_resized, data.gt_resized)

        for name in data.pred_raw:
            data.pred_resized[name] = net.resample(data.pred_raw[name],
                                                   width=width,
                                                   height=height,
                                                   reference=None,
                                                   type='LINEAR',
                                                   antialias=True)

        data.pred_final['flowL'] = net.scale(data.pred_resized['flowL'], (rescale_coeff_x, rescale_coeff_y))
        data.pred_final['flowR'] = net.scale(data.pred_resized['flowR'], (rescale_coeff_x, rescale_coeff_y))
        data.pred_final['disp0L'] = net.scale(data.pred_resized['disp0L'], (rescale_coeff_x))
        data.pred_final['disp1L'] = net.scale(data.pred_resized['disp1L'], (rescale_coeff_x))
        data.pred_final['dispChangeL'] = net.scale(data.pred_resized['dispChangeL'], (rescale_coeff_x))

        ## Connect L1 loss layers
        epe_losses = {}
        for name in data.pred_final:
            epe_losses[name] = \
              Layers.L1Loss(net,
                            (data.gt[name], data.pred_final[name]),
                            nout=1,
                            loss_weight=(1,), 
                            name='epe_' + name,
                            l2_per_location=True, 
                            normalize_by_num_entries=True, 
                            epsilon=0)
            epe_losses[name].enableOutput()

        epe_losses['flowL'].setName('flow_epe:L')
        epe_losses['flowR'].setName('flow_epe:R')
        epe_losses['disp0L'].setName('disp_epe:0L')
        epe_losses['disp1L'].setName('disp_epe:0R')
        epe_losses['dispChangeL'].setName('disp_change_err:L')

        data.copyNamesTo(net)

        return data.pred_final

    if generateNet:
        net = Network()
        Block(net)
        print net.toProto()

    return Block
