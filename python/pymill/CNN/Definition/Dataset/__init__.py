from pymill.CNN.Definition import Data as Data

class Dataset:
    def __init__(self, name, rendertype, phase):
        self._name = name
        self._rendertype = rendertype
        self._phase = phase

    def name(self): return self._name
    def rendertype(self): return self._rendertype

class SintelTrain(Dataset):
    def __init__(self, rendertype, phase):
        Dataset.__init__(self, 'sintel', rendertype, phase)

    def width(self): return 1024
    def height(self): return 436
    def meanColors(self):
        if self._rendertype == 'CLEAN': return (76.4783107737, 69.4660111681, 58.0279756163)
        else:                           return (91.2236713645, 82.6859238723, 69.5627393708)

    def dispLayer(self, net, **kwargs):
        kwargs['setting'] = 'DISPARITY_SINGLE'
        kwargs['rendertype'] = self._rendertype
        kwargs['phase'] = self._phase
        kwargs['collection_list'] = '/misc/lmbraid17/sceneflownet/common/data/4_bin-db/collection_lists/sintel_train.txt'
        kwargs['rand_permute'] = False
        return Data.BinaryData(net, **kwargs)

class FlyingStuff3DTest(Dataset):
    def __init__(self, rendertype, phase):
        Dataset.__init__(self, 'sintel', rendertype, phase)

    def width(self): return 960
    def height(self): return 540
    def meanColors(self):               # FIX ME FIX ME FIX ME FIX ME FIX ME FIX ME!!!
        if self._rendertype == 'CLEAN': return (76.4783107737, 69.4660111681, 58.0279756163)
        else:                           return (91.2236713645, 82.6859238723, 69.5627393708)

    def dispLayer(self, net, **kwargs):
        kwargs['setting'] = 'DISPARITY_SINGLE'
        kwargs['rendertype'] = self._rendertype
        kwargs['phase'] = self._phase
        kwargs['collection_list'] = '/misc/lmbraid17/sceneflownet/common/data/4_bin-db/collection_lists/v1/FlyingStuff3D_test.txt'
        kwargs['rand_permute'] = False
        return Data.BinaryData(net, **kwargs)

class MonkaaTest(Dataset):
    def __init__(self, rendertype, phase):
        Dataset.__init__(self, 'monkaa', rendertype, phase)

    def width(self): return 960
    def height(self): return 540
    def meanColors(self):               # FIX ME FIX ME FIX ME FIX ME FIX ME FIX ME!!!
        if self._rendertype == 'CLEAN': return (76.4783107737, 69.4660111681, 58.0279756163)
        else:                           return (91.2236713645, 82.6859238723, 69.5627393708)

    def dispLayer(self, net, **kwargs):
        kwargs['setting'] = 'DISPARITY_SINGLE'
        kwargs['rendertype'] = self._rendertype
        kwargs['phase'] = self._phase
        kwargs['collection_list'] = '/misc/lmbraid17/sceneflownet/common/data/4_bin-db/collection_lists/v1/monkaa_test.txt'
        kwargs['rand_permute'] = False
        return Data.BinaryData(net, **kwargs)

class Kitti2012Train(Dataset):
    def __init__(self, phase):
        Dataset.__init__(self, 'kitti2012', 'FINAL', phase)

    def width(self): return 1224
    def height(self): return 370
    def meanColors(self):               # FIX ME FIX ME FIX ME FIX ME FIX ME FIX ME!!!
        if self._rendertype == 'CLEAN': return (76.4783107737, 69.4660111681, 58.0279756163)
        else:                           return (91.2236713645, 82.6859238723, 69.5627393708)

    def dispLayer(self, net, **kwargs):
        kwargs['setting'] = 'DISPARITY_SINGLE'
        kwargs['rendertype'] = self._rendertype
        kwargs['phase'] = self._phase
        kwargs['collection_list'] = '/misc/lmbraid17/sceneflownet/common/data/4_bin-db/collection_lists/kitti2012_train.txt'
        kwargs['rand_permute'] = False
        return Data.BinaryData(net, **kwargs)

class Kitti2015Train(Dataset):
    def __init__(self, phase):
        Dataset.__init__(self, 'kitti2012', 'FINAL', phase)

    def width(self): return 1224
    def height(self): return 370
    def meanColors(self):               # FIX ME FIX ME FIX ME FIX ME FIX ME FIX ME!!!
        if self._rendertype == 'CLEAN': return (76.4783107737, 69.4660111681, 58.0279756163)
        else:                           return (91.2236713645, 82.6859238723, 69.5627393708)

    def dispLayer(self, net, **kwargs):
        kwargs['setting'] = 'DISPARITY_SINGLE'
        kwargs['rendertype'] = self._rendertype
        kwargs['phase'] = self._phase
        kwargs['collection_list'] = '/misc/lmbraid17/sceneflownet/common/data/4_bin-db/collection_lists/kitti2015_train.txt'
        kwargs['rand_permute'] = False
        return Data.BinaryData(net, **kwargs)

def get(name=None, rendertype=None, phase=None):
    if   name == 'sintel.train.clean':        return SintelTrain('CLEAN', phase)
    elif name == 'sintel.train.final':        return SintelTrain('FINAL', phase)
    elif name.startswith('kitti2012.train'):  return Kitti2012Train(phase)
    elif name.startswith('kitti2015.train'):  return Kitti2015Train(phase)
    elif name == 'FlyingStuff3D.test.clean':  return FlyingStuff3DTest('CLEAN', phase)
    elif name == 'FlyingStuff3D.test.final':  return FlyingStuff3DTest('FINAL', phase)
    elif name == 'monkaa.test.clean':         return MonkaaTest('CLEAN', phase)
    elif name == 'monkaa.test.final':         return MonkaaTest('FINAL', phase)
    else:
        raise Exception('unknown dataset: %s' % name)

