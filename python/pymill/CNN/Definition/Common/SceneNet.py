#!/usr/bin/python

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
    def Block(net, datasetName, output, prefix=None, use_augmentation_mean=True):
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
        img0L, img0R, img1L, img1R,  \
          flowL_gt, flowR_gt,        \
          disp0L_gt, disp1L_gt, dispChangeL_gt = dataset.flowLayer(net)
        ## Connect data to Deploy-mode net
        blobs_dict = {'img0L': img0L,
                      'img0R': img0R,
                      'img1L': img1L,
                      'img1R': img1R,
                      'flowL_gt': flowL_gt,
                      'flowR_gt': flowR_gt,
                      'disp0L_gt': disp0L_gt,
                      'disp1L_gt': disp1L_gt,
                      'dispChangeL_gt': dispChangeL_gt,
                     }
        ## TODO
        predictions = DeployBlock(net, blobs_dict,
                                  dataset.width(), dataset.height(),
                                  dataset.meanColors(), use_augmentation_mean)
        flowL_pred  = predictions.predict_flowL_final
        flowR_pred  = predictions.predict_flowR_final
        disp0L      = predictions.predict_disp0L_final
        disp1L      = predictions.predict_disp1L_final
        dispChangeL = predictions.predict_dispChangeL_final
        ## Output network input and output
        if output:
            if prefix:
                out_path = 'output_%s_%s' % (prefix, datasetName) 
            else:
                out_path = 'output_%s' % datasetName
            os.makedirs(out_path)

            ## Write configuration file for viewer tool
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
            f.write('1 2 -flowR_gt.flo\n')
            f.write('2 2 -dispChangeL_gt.float3\n')
            f.write('3 2 none\n')
            f.write('0 3 -flowL_pred.flo\n')
            f.write('1 3 -flowR_pred.flo\n')
            f.write('2 3 -dispChangeL_pred.float3\n')
            f.write('3 3 none\n')
            f.close()

            net.writeImage(img0L,     folder=out_path, prefix='', suffix='-img0L')
            net.writeImage(img0R,     folder=out_path, prefix='', suffix='-img0R')
            net.writeImage(img1L,     folder=out_path, prefix='', suffix='-img1L')
            net.writeImage(img1R,     folder=out_path, prefix='', suffix='-img1R')
            net.writeFlow(flowL_gt,   folder=out_path, prefix='', suffix='-flowL_gt')
            net.writeFlow(flowR_gt,   folder=out_path, prefix='', suffix='-flowR_gt')
            net.writeFloat(disp0L_gt, folder=out_path, prefix='', suffix='-disp0L_gt')
            net.writeFloat(disp1L_gt, folder=out_path, prefix='', suffix='-disp1L_gt')
            net.writeFloat(dispChangeL_gt, 
                           folder=out_path, prefix='', suffix='-dispChangeL_gt')
            
            net.writeFlow( flowL_pred,  
                           folder=out_path, prefix='', suffix='-flowL_pred')
            net.writeFlow( flowR_pred,  
                           folder=out_path, prefix='', suffix='-flowR_pred')
            net.writeFloat(disp0L_pred, 
                           folder=out_path, prefix='', suffix='-disp0L_pred')
            net.writeFloat(disp1L_pred, 
                           folder=out_path, prefix='', suffix='-disp1L_pred')
            net.writeFloat(dispChangeL_pred, 
                           folder=out_path, prefix='', suffix='-dispChangeL_pred')

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
          disp0L_gt, disp1L_gt, dispChangeL_gt = dataset.flowLayer(net)

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
        net.writeFloat(dispChangeL_gt, 
                                  folder=out_path, prefix='', suffix='-dispChangeL_gt')

    if generateNet:
        net = Network()

        dataset = str(param('dataset'))
        if dataset is None:
            raise Exception('please specify dataset=...')

        Block(net, dataset)

        print net.toProto()

    return Block


def standardDeploy(NetworkBlock, generateNet=True):
    '''
    @brief Create Deploy-mode wrapper for a raw net
    
    @param NetworkBlock Block(..) function of a raw network
    @param generateNet IFF TRUE, a network prototxt will be made and printed
    
    @returns Deploy-mode Block(..) function that adds data preprocessing and results postprocessing to a raw network
    '''
    def Block(net, input_blobs, width, height, mean_color, augmentation_mean=True):
        '''
        @brief Add data preprocessing to a network and connect data inputs
        
        @param net Incomplete network definition
        @param input_blobs Dictionary of input blobs (name->blob)
        @param width Input width (pixels)
        @param height Input height (pixels)
        @param mean_color Data mean to be subtracted from data if "augmentation_mean" is FALSE
        @param augmentation_mean IFF TRUE, data mean will be computed on the fly
        
        @returns Scene flow prediction layers of the network "net"
        '''
        blobs = net.namedBlobs()
        
        ## Connect inputs
        for key, value in input_blobs.iteritems():
            setattr(blobs, key, value)
        
        ## Input DATA
        data_blobs = ('img0L'. 'img0R', 'img1L', 'img1R')
        ## Input GROUNDTRUTH
        gt_blobs = ('flowL', 'flowR', 'disp0L', 'disp1L', 'dispChangeL')
        
        ## Rescale input images to [0,1]
        for blob_name in data_blobs
            #setattr(blobs, blob_name + 's', net.imageToRange01(getattr(blobs, blob_name)))
            scaled = net.imageToRange01(getattr(blobs, blob_name))
            scaled.setName(blob_name + 's')
        
        ## Subtract given mean or connect mean computation layer
        if augmentation_mean:
            for blob_name in data_blobs:
                setattr(blobs, 
                        blob_name + '_nomean',
                        net.subtractAugmentationMean(getattr(blobs, blob_name + 's'),
                                                     name=blob_name + "s_aug",
                                                     width=width, height=height))
        else:
            for blob_name in data_blobs:
                setattr(blobs, 
                        blob_name + '_nomean', 
                        net.subtractMean(getattr(blobs, blob_name + 's'), 
                                         mean_color))
                        
        ## Resample input data (needs to be 64-pixels aligned)
        divisor = 64.
        temp_width = ceil(width/divisor) * divisor
        temp_height = ceil(height/divisor) * divisor
        rescale_coeff_x = width / temp_width
        rescale_coeff_y = height / temp_height

        for blob_name in data_blobs:
            setattr(blobs, 
                    blob_name + '_nomean_resize', 
                    net.resample(getattr(blobs, blob_name + '_nomean'),
                                 width=temp_width, height=temp_height, 
                                 type='LINEAR', antialias=True))

        for blob_name in gt_blobs:
            # Use NEAREST here, since kitti gt is sparse
            setattr(blobs, 
                    blob_name + '_gt_resize', 
                    net.resample(blob_name, 
                                 width=temp_width, height=temp_height, 
                                 type='NEAREST', antialias=True) 
                 
        net_blobs = {}
        for blob_name in data_blobs:
            net_blobs[blob_name + '_nomean_resize'] = \
              getattr(blobs, blob_name + '_nomean_resize')
        for blob_name in gt_blobs:
            net_blobs[blob_name + '_gt_resize'] = \
              getattr(blobs, blob_name + '_gt_resize')

        ## Connect data preprocessing layers to raw net
        from net import Block as Network
        prediction = NetworkBlock(net, net_blobs)

        for blob_name in gt_blobs:
            setattr(blobs, 
                    'predict_' + blob_name + '_resize',
                    net.resample(prediction[blob_name], 
                                 width=width, height=height, 
                                 reference=None, type='LINEAR', antialias=True)
            setattr(blobs, 
                    'predict_' + blob_name + '_final', 
                    net.scale(getattr(blobs, 'predict_' + blob_name + '_resize'),
                              (rescale_coeff_x, rescale_coeff_y))

        ## Connect L1 loss layers
        epe_losses = {}
        for blob_name in gt_blobs:
            epe_losses[blob_name] = \
              Layers.L1Loss(net,
                            (getattr(blobs,blob_name), 
                            getattr(blobs,'predict_' + blob_name + '_final')), 
                            nout=1, 
                            loss_weight=(1,), 
                            name='EPE_' + blob_name, 
                            l2_per_location=True, 
                            normalize_by_num_entries=True, 
                            epsilon=0)
            epe_losses[blob_name].setName('epe_'+blob_name)
            epe_losses[blob_name].enableOutput()

        return tuple([getattr(blobs, 'predict_'+blob_name+'_final') 
                      for blob_name in gt_blobs])

    if generateNet:
        net = Network()
        Block(net)
        print net.toProto()

    return Block
