#!/usr/bin/python

##
# Optical flow
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
        img0, img1, flow_gt = dataset.flowLayer(net)
        ## Connect data to Deploy-mode net
        flow_pred = DeployBlock(net, img0, img1, flow_gt, 
                                dataset.width(), dataset.height(),
                                dataset.meanColors(), use_augmentation_mean)
        ## Output network input and output
        if output:
            if prefix:
                out_path = 'output_%s_%s' % (prefix, datasetName)
            else:
                out_path = 'output_%s' % datasetName
            if not os.path.isdir(out_path):
                os.makedirs(out_path)
            
            ## Write configuration file for viewer tool
            f = open('%s/viewer.cfg' % out_path, 'w')
            f.write('3 2\n')
            f.write('0 0 -img0.ppm\n')
            f.write('1 0 -img1.ppm\n')
            f.write('2 0 EPE(-flow.flo,-gt.flo)\n')
            f.write('0 1 -flow.flo\n')
            f.write('1 1 -gt.flo\n')
            f.write('2 1 DIFF(-flow.flo,-gt.flo)\n')
            f.close()
            
            ## Create network file outputs
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
        img0, img1, flow_gt = dataset.flowLayer(net)

        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        ## Write configuration file for viewer tool
        f = open('%s/viewer.cfg' % (out_path), 'w')
        f.write('2 2\n')
        f.write('0 0 -img0.ppm\n')
        f.write('1 0 -img1.ppm\n')
        f.write('0 1 -gt.flo\n')
        f.write('1 1 none\n')
        f.close()
        
        ## Create network file outputs
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
    '''
    @brief Create Deploy-mode wrapper for a raw net
    
    @param NetworkBlock Block(..) function of a raw network
    @param generateNet IFF TRUE, a network prototxt will be made and printed
    
    @returns Deploy-mode Block(..) function that adds data preprocessing and results postprocessing to a raw network
    '''
    def Block(net, img0, img1, flow_gt, 
              width, height, mean_color, augmentation_mean=True):
        '''
        @brief Add data preprocessing to a network and connect data inputs
        
        @param net Incomplete network definition
        @param img0 Optical flow: First image
        @param img1 Optical flow: Second image
        @param flow_gt Optical flow: Flow groundtruth
        @param width Input width (pixels)
        @param height Input height (pixels)
        @param mean_color Data mean to be subtracted from data if "augmentation_mean" is FALSE
        @param augmentation_mean IFF TRUE, data mean will be computed on the fly
        
        @returns Optical flow prediction layer of the network "net"
        '''
        blobs = net.namedBlobs()
        ## Connect inputs
        blobs.img0 = img0
        blobs.img1 = img1
        blobs.flow_gt = flow_gt
        ## Rescale input images to [0,1]
        blobs.img0s = net.imageToRange01(blobs.img0)
        blobs.img1s = net.imageToRange01(blobs.img1)

        ## Subtract given mean or connect mean computation layer
        if augmentation_mean:
            blobs.img0_nomean = net.subtractAugmentationMean(blobs.img0s, name="img0s_aug", width=width, height=height)
            blobs.img1_nomean = net.subtractAugmentationMean(blobs.img1s, name="img1s_aug", width=width, height=height)
        else:
            blobs.img0_nomean = net.subtractMean(blobs.img0s, mean_color)
            blobs.img1_nomean = net.subtractMean(blobs.img1s, mean_color)
            
        ## Resample input data (needs to be 64-pixels aligned)
        divisor = 64.
        temp_width = ceil(width/divisor) * divisor
        temp_height = ceil(height/divisor) * divisor
        rescale_coeff_x = width / temp_width
        rescale_coeff_y = height / temp_height

        blobs.img0_nomean_resize = net.resample(blobs.img0_nomean, 
                                                width=temp_width, 
                                                height=temp_height, 
                                                type='LINEAR', 
                                                antialias=True)
        blobs.img1_nomean_resize = net.resample(blobs.img1_nomean, 
                                                width=temp_width, 
                                                height=temp_height, 
                                                type='LINEAR', 
                                                antialias=True)
        ## Use NEAREST here, since KITTI groundtruth is sparse
        blobs.flow_gt_resize     = net.resample(blobs.flow_gt,     
                                                width=temp_width, 
                                                height=temp_height, 
                                                type='NEAREST', 
                                                antialias=True)
        
        ## Connect data preprocessing layers to raw net
        from net import Block as Network
        prediction = NetworkBlock(net,
                             blobs.img0_nomean_resize,
                             blobs.img1_nomean_resize,
                             blobs.flow_gt_resize)
        
        ## Resample net output to input resolution
        blobs.predict_flow_resize = net.resample(prediction, 
                                                 width=width, 
                                                 height=height, 
                                                 reference=None, 
                                                 type='LINEAR', 
                                                 antialias=True)
        blobs.predict_flow_final  = net.scale(blobs.predict_flow_resize,
                                              (rescale_coeff_x, rescale_coeff_y))
        
        ## Connect L1 flow loss layer
        epe_loss = Layers.L1Loss(net, (blobs.flow_gt, blobs.predict_flow_final),
                                 nout=1, loss_weight=(1,), name='flow_epe',
                                 l2_per_location=True, 
                                 normalize_by_num_entries=True, epsilon=0)
        epe_loss.setName('flow_epe')
        epe_loss.enableOutput()

        return blobs.predict_flow_final

    if generateNet:
        net = Network()
        Block(net)
        print net.toProto()

    return Block
