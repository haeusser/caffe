from pymill.CNN.Definition import Data as Data

##
# Dataset helpers: 
# - Get available datasets
# - Dataset helpers which can report data dimensions and mean colors, and 
#   can setup data input layers for networks for specified task settings
#   (disparity, optical flow, scene flow)
##


## Root folder with copies of all the dataset collection lists
COLL_LISTS_DIR = '/misc/lmbraid17/sceneflownet/common/data/4_bin-db/collection_lists'


class Dataset:
    '''
    @brief Base class. Every dataset is a separate class and inherits from this.
    '''
    def __init__(self, name, rendertype, phase):
        self._name = name
        self._rendertype = rendertype
        self._phase = phase

    def name(self): 
        '''@brief Get this dataset's name'''
        return self._name
      
    def rendertype(self): 
        '''@brief Get this dataset's render type ('CLEAN'/'FINAL')'''
        return self._rendertype
  
    def width(self): 
        '''@brief Image width (in pixels) for this dataset'''
        raise Exception('Not available/implemented for this dataset')
    def height(self): 
        '''@brief Image height (in pixels) for this dataset'''
        raise Exception('Not available/implemented for this dataset')
      
    def meanColors(self):
        '''@brief Get mean color vector for this dataset'''
        raise Exception('Not available/implemented for this dataset')
    
    def dispLayer(self, net, **kwarsgs):
        '''@brief Setup a disparity data layer for a network'''
        raise Exception('Not available/implemented for this dataset')
    def flowLayer(self, net, **kwarsgs):
        '''@brief Setup an optical flow data layer for a network'''
        raise Exception('Not available/implemented for this dataset')
    def sceneFlowLayer(self, net, **kwarsgs):
        '''@brief Setup a scene flow data layer for a network'''
        raise Exception('Not available/implemented for this dataset')


class SintelTrain(Dataset):
    '''@brief SINTEL training dataset (disparity, optical flow)'''
    def __init__(self, rendertype, phase):
        Dataset.__init__(self, 'sintel', rendertype, phase)

    def width(self): return 1024
    def height(self): return 436
    def meanColors(self):
        if self._rendertype == 'CLEAN': 
            return (76.4783107737, 69.4660111681, 58.0279756163)
        else:                           
            return (91.2236713645, 82.6859238723, 69.5627393708)

    def dispLayer(self, net, **kwargs):
        kwargs['setting']         = 'DISPARITY_SINGLE'
        kwargs['rendertype']      = self._rendertype
        kwargs['phase']           = self._phase
        kwargs['collection_list'] = COLL_LISTS_DIR+'/sintel_train.txt'
        kwargs['rand_permute']    = False
        return Data.BinaryData(net, **kwargs)

    def flowLayer(self, net, **kwargs):
        kwargs['setting']         = 'OPTICAL_FLOW_SINGLE'
        kwargs['rendertype']      = self._rendertype
        kwargs['phase']           = self._phase
        kwargs['collection_list'] = COLL_LISTS_DIR+'/sintel_train.txt'
        kwargs['rand_permute']    = False
        return Data.BinaryData(net, **kwargs)


class FlyingStuff3DTest(Dataset):
    '''@brief FlyingStuff3D testing dataset (disparity, optical flow, scene flow)'''
    def __init__(self, rendertype, phase):
        Dataset.__init__(self, 'FlyingStuff3D', rendertype, phase)

    def width(self): return 960
    def height(self): return 540
    def meanColors(self):               # FIX ME FIX ME FIX ME FIX ME FIX ME FIX ME!!!
        if self._rendertype == 'CLEAN': 
            return (76.4783107737, 69.4660111681, 58.0279756163)
        else:                           
            return (91.2236713645, 82.6859238723, 69.5627393708)

    def dispLayer(self, net, **kwargs):
        kwargs['setting']         = 'DISPARITY_SINGLE'
        kwargs['rendertype']      = self._rendertype
        kwargs['phase']           = self._phase
        kwargs['collection_list'] = COLL_LISTS_DIR+'/v1/FlyingStuff3D_test.txt'
        kwargs['rand_permute']    = False
        return Data.BinaryData(net, **kwargs)

    def flowLayer(self, net, **kwargs):
        kwargs['setting']         = 'OPTICAL_FLOW_SINGLE'
        kwargs['rendertype']      = self._rendertype
        kwargs['phase']           = self._phase
        kwargs['collection_list'] = COLL_LISTS_DIR+'/v1/FlyingStuff3D_test.txt'
        kwargs['rand_permute']    = False
        return Data.BinaryData(net, **kwargs)
      
    def sceneFlowLayer(self, net, **kwargs):
        kwargs['setting']         = 'SCENE_FLOW_SINGLE'
        kwargs['rendertype']      = self._rendertype
        kwargs['phase']           = self._phase
        kwargs['collection_list'] = COLL_LISTS_DIR+'/v1/FlyingStuff3D_test.txt'
        kwargs['rand_permute']    = False
        return Data.BinaryData(net, **kwargs)


class FlyingStuff3DNewTest(Dataset):
    '''@brief FlyingStuff3D testing dataset (disparity, optical flow, scene flow)'''
    def __init__(self, rendertype, phase):
        Dataset.__init__(self, 'FlyingStuff3D_new', rendertype, phase)

    def width(self): return 960
    def height(self): return 540
    def meanColors(self):               # FIX ME FIX ME FIX ME FIX ME FIX ME FIX ME!!!
        if self._rendertype == 'CLEAN':
            return (76.4783107737, 69.4660111681, 58.0279756163)
        else:
            return (91.2236713645, 82.6859238723, 69.5627393708)

    def dispLayer(self, net, **kwargs):
        kwargs['setting']         = 'DISPARITY_SINGLE'
        kwargs['rendertype']      = self._rendertype
        kwargs['phase']           = self._phase
        kwargs['collection_list'] = COLL_LISTS_DIR+'/v1/FlyingStuff3D_new_test.txt'
        kwargs['rand_permute']    = False
        return Data.BinaryData(net, **kwargs)

    def flowLayer(self, net, **kwargs):
        kwargs['setting']         = 'OPTICAL_FLOW_SINGLE'
        kwargs['rendertype']      = self._rendertype
        kwargs['phase']           = self._phase
        kwargs['collection_list'] = COLL_LISTS_DIR+'/v1/FlyingStuff3D_new_WebP_test.txt' #FlyingStuff3D_new_test.txt'
        kwargs['rand_permute']    = False
        return Data.BinaryData(net, **kwargs)

    def sceneFlowLayer(self, net, **kwargs):
        kwargs['setting']         = 'SCENE_FLOW_SINGLE'
        kwargs['rendertype']      = self._rendertype
        kwargs['phase']           = self._phase
        kwargs['collection_list'] = COLL_LISTS_DIR+'/v1/FlyingStuff3D_new_test.txt'
        kwargs['rand_permute']    = False
        return Data.BinaryData(net, **kwargs)
      
      

class FakeKittiTrees(Dataset):
    '''@brief FlyingStuff3D testing dataset (disparity, optical flow, scene flow)'''
    def __init__(self, rendertype, phase):
        Dataset.__init__(self, 'FakeKittiTrees', rendertype, phase)

    def width(self): return 960
    def height(self): return 540
    def meanColors(self):               # FIX ME FIX ME FIX ME FIX ME FIX ME FIX ME!!!
        if self._rendertype == 'CLEAN': 
            return (76.4783107737, 69.4660111681, 58.0279756163)
        else:                           
            return (91.2236713645, 82.6859238723, 69.5627393708)

    def dispLayer(self, net, **kwargs):
        kwargs['setting']         = 'DISPARITY_SINGLE'
        kwargs['rendertype']      = self._rendertype
        kwargs['phase']           = self._phase
        kwargs['collection_list'] = COLL_LISTS_DIR+'/v1/FakeKittiTrees_full.txt'
        kwargs['rand_permute']    = False
        return Data.BinaryData(net, **kwargs)

    def flowLayer(self, net, **kwargs):
        kwargs['setting']         = 'OPTICAL_FLOW_SINGLE'
        kwargs['rendertype']      = self._rendertype
        kwargs['phase']           = self._phase
        kwargs['collection_list'] = COLL_LISTS_DIR+'/v1/FakeKittiTrees_full.txt'
        kwargs['rand_permute']    = False
        return Data.BinaryData(net, **kwargs)
      
    def sceneFlowLayer(self, net, **kwargs):
        kwargs['setting']         = 'SCENE_FLOW_SINGLE'
        kwargs['rendertype']      = self._rendertype
        kwargs['phase']           = self._phase
        kwargs['collection_list'] = COLL_LISTS_DIR+'/v1/FakeKittiTrees_full.txt'
        kwargs['rand_permute']    = False
        return Data.BinaryData(net, **kwargs)

class MonkaaTest(Dataset):
    '''@brief Monkaa testing dataset (disparity, optical flow, scene flow)'''
    def __init__(self, rendertype, phase):
        Dataset.__init__(self, 'monkaa', rendertype, phase)

    def width(self): return 960
    def height(self): return 540
    def meanColors(self):               # FIX ME FIX ME FIX ME FIX ME FIX ME FIX ME!!!
        if self._rendertype == 'CLEAN': 
            return (76.4783107737, 69.4660111681, 58.0279756163)
        else:                           
            return (91.2236713645, 82.6859238723, 69.5627393708)

    def dispLayer(self, net, **kwargs):
        kwargs['setting']         = 'DISPARITY_SINGLE'
        kwargs['rendertype']      = self._rendertype
        kwargs['phase']           = self._phase
        kwargs['collection_list'] = COLL_LISTS_DIR+'/v1/monkaa_test.txt'
        kwargs['rand_permute']    = False
        return Data.BinaryData(net, **kwargs)

    def flowLayer(self, net, **kwargs):
        kwargs['setting']         = 'OPTICAL_FLOW_SINGLE'
        kwargs['rendertype']      = self._rendertype
        kwargs['phase']           = self._phase
        kwargs['collection_list'] = COLL_LISTS_DIR+'/v1/monkaa_test.txt'
        kwargs['rand_permute']    = False
        return Data.BinaryData(net, **kwargs)
      
    def sceneFlowLayer(self, net, **kwargs):
        kwargs['setting']         = 'SCENE_FLOW_SINGLE'
        kwargs['rendertype']      = self._rendertype
        kwargs['phase']           = self._phase
        kwargs['collection_list'] = COLL_LISTS_DIR+'/v1/monkaa_test.txt'
        kwargs['rand_permute']    = False
        return Data.BinaryData(net, **kwargs)
      
class MonkaaRelease(Dataset):
    '''@brief Monkaa testing dataset (disparity, optical flow, scene flow)'''
    def __init__(self, rendertype, phase):
        Dataset.__init__(self, 'monkaa.release', rendertype, phase)

    def width(self): return 960
    def height(self): return 540
    def meanColors(self):               # FIX ME FIX ME FIX ME FIX ME FIX ME FIX ME!!!
        if self._rendertype == 'CLEAN': 
            return (76.4783107737, 69.4660111681, 58.0279756163)
        else:                           
            return (91.2236713645, 82.6859238723, 69.5627393708)

    def dispLayer(self, net, **kwargs):
        kwargs['setting']         = 'DISPARITY_SINGLE'
        kwargs['rendertype']      = self._rendertype
        kwargs['phase']           = self._phase
        kwargs['collection_list'] = COLL_LISTS_DIR+'/v1/monkaa_release.txt'
        kwargs['rand_permute']    = False
        return Data.BinaryData(net, **kwargs)

    def flowLayer(self, net, **kwargs):
        kwargs['setting']         = 'OPTICAL_FLOW_SINGLE'
        kwargs['rendertype']      = self._rendertype
        kwargs['phase']           = self._phase
        kwargs['collection_list'] = COLL_LISTS_DIR+'/v1/monkaa_release.txt'
        kwargs['rand_permute']    = False
        return Data.BinaryData(net, **kwargs)
      
    def sceneFlowLayer(self, net, **kwargs):
        kwargs['setting']         = 'SCENE_FLOW_SINGLE'
        kwargs['rendertype']      = self._rendertype
        kwargs['phase']           = self._phase
        kwargs['collection_list'] = COLL_LISTS_DIR+'/v1/monkaa_release.txt'
        kwargs['rand_permute']    = False
        return Data.BinaryData(net, **kwargs)


class Kitti2012Train(Dataset):
    '''@brief 2012 KITTI training dataset (disparity, optical flow)'''
    def __init__(self, phase):
        Dataset.__init__(self, 'kitti2012', 'FINAL', phase)

    def width(self): return 1224
    def height(self): return 370
    def meanColors(self):               # FIX ME FIX ME FIX ME FIX ME FIX ME FIX ME!!!
        if self._rendertype == 'CLEAN': 
            return (76.4783107737, 69.4660111681, 58.0279756163)
        else:                           
            return (91.2236713645, 82.6859238723, 69.5627393708)

    def dispLayer(self, net, **kwargs):
        kwargs['setting']         = 'DISPARITY_SINGLE'
        kwargs['rendertype']      = self._rendertype
        kwargs['phase']           = self._phase
        kwargs['collection_list'] = COLL_LISTS_DIR+'/kitti2012_train.txt'
        kwargs['rand_permute']    = False
        return Data.BinaryData(net, **kwargs)

    def flowLayer(self, net, **kwargs):
        kwargs['setting']         = 'OPTICAL_FLOW_SINGLE'
        kwargs['rendertype']      = self._rendertype
        kwargs['phase']           = self._phase
        kwargs['collection_list'] = COLL_LISTS_DIR+'/kitti2012_train.txt'
        kwargs['rand_permute']    = False
        return Data.BinaryData(net, **kwargs)


class Kitti2015Train(Dataset):
    '''@brief 2015 KITTI training dataset (disparity, optical flow, scene flow)'''
    def __init__(self, phase):
        Dataset.__init__(self, 'kitti2015', 'FINAL', phase)

    def width(self): return 1224
    def height(self): return 370
    def meanColors(self):               # FIX ME FIX ME FIX ME FIX ME FIX ME FIX ME!!!
        if self._rendertype == 'CLEAN': 
            return (76.4783107737, 69.4660111681, 58.0279756163)
        else:                           
            return (91.2236713645, 82.6859238723, 69.5627393708)

    def dispLayer(self, net, **kwargs):
        kwargs['setting']         = 'DISPARITY_SINGLE'
        kwargs['rendertype']      = self._rendertype
        kwargs['phase']           = self._phase
        kwargs['collection_list'] = COLL_LISTS_DIR+'/kitti2015_train.txt'
        kwargs['rand_permute']    = False
        return Data.BinaryData(net, **kwargs)

    def flowLayer(self, net, **kwargs):
        kwargs['setting']         = 'OPTICAL_FLOW_SINGLE'
        kwargs['rendertype']      = self._rendertype
        kwargs['phase']           = self._phase
        kwargs['collection_list'] = COLL_LISTS_DIR+'/kitti2015_train.txt'
        kwargs['rand_permute']    = False
        return Data.BinaryData(net, **kwargs)
    
    #def sceneFlowLayer(self, net, **kwargs):
        ## TODO


class FlyingChairsValidation(Dataset):
    '''@brief FlyingChairs (FlowNet) validation dataset (optical flow)'''
    def __init__(self, phase):
        Dataset.__init__(self, 'chairs.val', 'CLEAN', phase)

    def width(self): return 512
    def height(self): return 384
    def meanColors(self):               # FIX ME FIX ME FIX ME FIX ME FIX ME FIX ME!!!
        if self._rendertype == 'CLEAN': 
            return (76.4783107737, 69.4660111681, 58.0279756163)
        else:                           
            return (91.2236713645, 82.6859238723, 69.5627393708)

    def flowLayer(self, net, **kwargs):
        img0, img1, flow_gt, occ = Data.FlyingChairs(net,
            subset='TEST',
            phase=self._phase,
            **kwargs
        )
        return img0, img1, flow_gt


def get(name=None, rendertype=None, phase=None):
    '''
    @brief Get access to dataset information and data layer helper functions
    
    @param name Unique identifier for the requested dataset
    @param rendertype One of ['CLEAN', 'FINAL']
    @param phase One of ['TRAIN', 'TEST']
    
    @returns A Dataset instance for the requested dataset
    '''
    if   name == 'sintel.train.clean':        return SintelTrain('CLEAN', phase)
    elif name == 'sintel.train.final':        return SintelTrain('FINAL', phase)
    elif name.startswith('kitti2012.train'):  return Kitti2012Train(phase)
    elif name.startswith('kitti2015.train'):  return Kitti2015Train(phase)
    elif name == 'FlyingStuff3D.test.clean':  return FlyingStuff3DTest('CLEAN', phase)
    elif name == 'FlyingStuff3D.test.final':  return FlyingStuff3DTest('FINAL', phase)
    elif name == 'FlyingStuff3D_new.test.clean':  return FlyingStuff3DNewTest('CLEAN', phase)
    elif name == 'FlyingStuff3D_new.test.final':  return FlyingStuff3DNewTest('FINAL', phase)
    elif name == 'monkaa.test.clean':         return MonkaaTest('CLEAN', phase)
    elif name == 'monkaa.test.final':         return MonkaaTest('FINAL', phase)
    elif name == 'monkaa.release.clean':      return MonkaaRelease('CLEAN', phase)
    elif name == 'monkaa.release.final':      return MonkaaRelease('FINAL', phase)
    elif name == 'FakeKittiTrees.clean':      return FakeKittiTrees('CLEAN', phase)
    elif name == 'FakeKittiTrees.final':      return FakeKittiTrees('FINAL', phase)
    elif name == 'chairs.val':                return FlyingChairsValidation(phase)
    else:
        raise Exception('unknown dataset "%s" for phase "%s"' % (name, phase))


def getDatasetNames(task):
    '''
    @brief List available _TEST_ datasets for a given task
    
    @param task One of 'disp' (disparity), 'flow' (optical flow), 'sceneflow' (scene flow: disparity, disparity change, optical flow)
    
    @returns A tuple with the names of all datasets that contain necessary data for the specified task.
    '''
    ## Disparity
    if task == 'disp':
        return ('sintel.train.clean',
                'sintel.train.final',
                'monkaa.test.clean',
                'monkaa.test.final',
                'monkaa.release.clean',
                'monkaa.release.final',
                'FakeKittiTrees.clean',
                'FakeKittiTrees.final',
                'FlyingStuff3D.test.clean',
                'FlyingStuff3D.test.final',
                'FlyingStuff3D_new.test.clean',
                'FlyingStuff3D_new.test.final',
                'kitti2012.train',
                'kitti2015.train',)
    ## Optical flow
    elif task == 'flow':
        return ('sintel.train.clean',
                'sintel.train.final',
                'monkaa.test.clean',
                'monkaa.test.final',
                'monkaa.release.clean',
                'monkaa.release.final',
                'FakeKittiTrees.clean',
                'FakeKittiTrees.final',
                'FlyingStuff3D.test.clean',
                'FlyingStuff3D.test.final',
                'FlyingStuff3D_new.test.clean',
                'FlyingStuff3D_new.test.final',
                'kitti2012.train',
                'kitti2015.train',
                'chairs.val',)
    ## Scene flow
    elif task == 'sceneflow':
        return ('monkaa.test.clean',
                'monkaa.test.final',
                'monkaa.release.clean',
                'monkaa.release.final',
                'FakeKittiTrees.clean',
                'FakeKittiTrees.final',
                'FlyingStuff3D.test.clean',
                'FlyingStuff3D.test.final',
                'FlyingStuff3D_new.test.clean',
                'FlyingStuff3D_new.test.final',
                'kitti2015.train',)
    else:
        raise Exception('unknown task: "%s"' % task)