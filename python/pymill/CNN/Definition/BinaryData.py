#!/usr/bin/python

from CaffeAdapter import  *
from caffe.proto import caffe_pb2 as Proto
from pymill import Toolbox as tb
import os
from collections import OrderedDict

BIN_DB_DIR   = '/misc/lmbraid17/sceneflownet/common/data/4_bin-db'
SSD_BIN_DB_DIR   = '/misc/scratch0/hackathon/data/4_bin-db'

COLLECTIONLIST_DIR = '/misc/lmbraid17/sceneflownet/common/data/4_bin-db/collection_lists'


## Assemble samples to be read from BinaryDB
def Entry(name, offset):
  '''
  @brief Create a Protobuf DataEntry message instance
  @param name Name of the entry
  @param offset Offset of the entry, measured in whole frames relative to the current frame
  @returns A DataEntry message
  '''
  if not name:
    raise ValueError('>name< must be set')
  return Proto.DataEntry(name=name, offset=offset)


def Sample(entries):
  '''
  @brief Create a Protobuf DataSample message instance
  @param entries A tuple of DataEntry messages
  @returns A DataSample message
  '''
  if not entries or not isinstance(entries, tuple):
    raise ValueError('>entries< must be a tuple')
  return Proto.DataSample(entry=entries)


def NumberOfEntries(samples):
  '''
  @brief Extract the number of entries in a DataSample definition
  @warning This is super hacky and probably extremely dependent on the Protobuf version
  @param samples A DataSample or an iterable of DataSample
  @returns The number of entries in the first sample
  '''
  if isinstance(samples, tuple) or isinstance(samples, list):
    if samples:
      return len(samples[0].ListFields()[0][1])
    else:
      raise ValueError('>samples< must not be empty')
  elif isinstance(samples, Proto.DataSample):
    return len(samples.ListFields()[0][1])
  else:
    raise ValueError('>samples< must be a DataSample or an iterable of DataSample')


def DataParams(samples, bin_db_dir, collection_list_dir, collection_list, batch_size, verbose, rand_permute, rand_permute_seed=None, **kwargs):
  '''
  @brief Create a Protobuf DataParameter message instance for BinaryDB
  @param samples A tuple of DataSample messages
  @returns A DataParameter message with BinaryDB-specific entries
  '''
  params = {'source'    : bin_db_dir,
            'collection_list' : os.path.join(collection_list_dir, collection_list),
            'backend'   : Params.Data.BINARYDBWEBP,
            'batch_size': batch_size,
            'sample'    : samples,
            'verbose'   : verbose,
            'rand_permute'     : rand_permute}
  
  def use(p):
    if p in kwargs:
      params[p] = kwargs[p]

  use('prefetch')
  use('disk_reader_threads')
  use('error_based_sampling')
  use('sampling_alpha')
  use('sampling_beta')
  use('sampling_gamma')

  if rand_permute_seed is not None:
      params['rand_permute_seed'] = rand_permute_seed

  return params


def BinaryData_OpticalFlow(net, **kwargs):
  '''
  @brief Setup network inputs for optical flow
  @returns A list of single-blob network INPUT and LABEL
  '''
  samples = []

  if kwargs['rendertype'] == 'CLEAN' or kwargs['rendertype'] == 'BOTH':
      samples += (Sample((Entry('cleanImageL',  0),
                          Entry('cleanImageL', +1),
                          Entry('forwardFlowL',  0))),
                  Sample((Entry('cleanImageL',  0),
                          Entry('cleanImageL', -1),
                          Entry('backwardFlowL', 0))),
                  Sample((Entry('cleanImageR',  0),
                          Entry('cleanImageR', +1),
                          Entry('forwardFlowR',  0))),
                  Sample((Entry('cleanImageR',  0),
                          Entry('cleanImageR', -1),
                          Entry('backwardFlowR', 0))))

  if kwargs['rendertype'] == 'FINAL' or kwargs['rendertype'] == 'BOTH':
      samples += (Sample((Entry('finalImageL',  0),
                          Entry('finalImageL', +1),
                          Entry('forwardFlowL',  0))),
                  Sample((Entry('finalImageL',  0),
                          Entry('finalImageL', -1),
                          Entry('backwardFlowL', 0))),
                  Sample((Entry('finalImageR',  0),
                          Entry('finalImageR', +1),
                          Entry('forwardFlowR',  0))),
                  Sample((Entry('finalImageR',  0),
                          Entry('finalImageR', -1),
                          Entry('backwardFlowR', 0))))

  nout = NumberOfEntries(samples)
  if 'output_index' in kwargs:
      nout += 1
      del kwargs['output_index']

  return Layers.BinaryData(net,
                           nout=nout,
                           include=(Proto.NetStateRule(phase=kwargs['phase']),),
                           data_param=DataParams(samples, **kwargs))

def BinaryData_OpticalFlow_Single(net, **kwargs):
  '''
  @brief Setup network inputs for optical flow
  @returns A list of single-blob network INPUT and LABEL
  '''
  samples = []

  if kwargs['rendertype'] == 'CLEAN' or kwargs['rendertype'] == 'BOTH':
      samples += (Sample((Entry('cleanImageL',  0),
                          Entry('cleanImageL', +1),
                          Entry('forwardFlowL',  0))),)

  if kwargs['rendertype'] == 'FINAL' or kwargs['rendertype'] == 'BOTH':
      samples += (Sample((Entry('finalImageL',  0),
                          Entry('finalImageL', +1),
                          Entry('forwardFlowL',  0))),)

  nout = NumberOfEntries(samples)
  if 'output_index' in kwargs:
      nout += 1
      del kwargs['output_index']

  return Layers.BinaryData(net,
                           nout=nout,
                           include=(Proto.NetStateRule(phase=kwargs['phase']),),
                           data_param=DataParams(samples, **kwargs))


def _dataStructFromSceneFlow(list):
    struct = DataStruct()

    struct.add('inp', '')

    struct.inp.img0L      = list[0]
    struct.inp.img0R      = list[1]
    struct.inp.img1L      = list[2]
    struct.inp.img1R      = list[3]
    struct.gt.flowL       = list[4]
    struct.gt.flowR       = list[5]
    struct.gt.disp0L      = list[6]
    struct.gt.disp1L      = list[7]
    struct.gt.dispChangeL = list[8]

    return struct

def _dataStructFromSceneFlow2(list):
    struct = DataStruct()

    struct.add('inp', '')

    struct.inp.img0L      = list[0]
    struct.inp.img0R      = list[1]
    struct.inp.img1L      = list[2]
    struct.inp.img1R      = list[3]
    struct.gt.flowL       = list[4]
    struct.gt.disp0L      = list[5]
    struct.gt.dispChangeL = list[6]

    return struct


def BinaryData_SceneFlow(net, **kwargs):
  '''
  @brief Setup network inputs for scene flow
  @returns A list of single-blob network INPUT and LABEL
  '''
  samples = []

  if kwargs['rendertype'] == 'CLEAN' or kwargs['rendertype'] == 'BOTH':
      samples += (Sample((Entry('cleanImageL',  0),
                          Entry('cleanImageR',  0),
                          Entry('cleanImageL', +1),
                          Entry('cleanImageR', +1),
                          Entry('forwardFlowL',  0),
                          Entry('forwardFlowR',  0),
                          Entry('dispL',  0),
                          Entry('dispL', +1),
                          Entry('forwardDispChangeL', 0))),
                  Sample((Entry('cleanImageL',  0),
                          Entry('cleanImageR',  0),
                          Entry('cleanImageL', -1),
                          Entry('cleanImageR', -1),
                          Entry('backwardFlowL',  0),
                          Entry('backwardFlowR',  0),
                          Entry('dispL',  0),
                          Entry('dispL', -1),
                          Entry('backwardDispChangeL', 0))))

  if kwargs['rendertype'] == 'FINAL' or kwargs['rendertype'] == 'BOTH':
      samples += (Sample((Entry('finalImageL',  0),
                          Entry('finalImageR',  0),
                          Entry('finalImageL', +1),
                          Entry('finalImageR', +1),
                          Entry('forwardFlowL',  0),
                          Entry('forwardFlowR',  0),
                          Entry('dispL',  0),
                          Entry('dispL', +1),
                          Entry('forwardDispChangeL', 0))),
                  Sample((Entry('finalImageL',  0),
                          Entry('finalImageR',  0),
                          Entry('finalImageL', -1),
                          Entry('finalImageR', -1),
                          Entry('backwardFlowL',  0),
                          Entry('backwardFlowR',  0),
                          Entry('dispL',  0),
                          Entry('dispL', -1),
                          Entry('backwardDispChangeL', 0))))

  nout = NumberOfEntries(samples)
  if 'output_index' in kwargs:
      nout += 1
      del kwargs['output_index']

  blobs = Layers.BinaryData(net,
                           nout=nout,
                           include=(Proto.NetStateRule(phase=kwargs['phase']),),
                           data_param=DataParams(samples, **kwargs))

  if kwargs['return_struct']: return _dataStructFromSceneFlow(blobs)
  else: return blobs


def BinaryData_SceneFlow_Single(net, **kwargs):
  '''
  @brief Setup network inputs for scene flow
  @returns A list of single-blob network INPUT and LABEL
  '''
  samples = []

  if kwargs['rendertype'] == 'CLEAN' or kwargs['rendertype'] == 'BOTH':
      samples += (Sample((Entry('cleanImageL',  0),
                          Entry('cleanImageR',  0),
                          Entry('cleanImageL', +1),
                          Entry('cleanImageR', +1),
                          Entry('forwardFlowL',  0),
                          Entry('forwardFlowR',  0),
                          Entry('dispL',  0),
                          Entry('dispL', +1),
                          Entry('forwardDispChangeL', 0))),)

  if kwargs['rendertype'] == 'FINAL' or kwargs['rendertype'] == 'BOTH':
      samples += (Sample((Entry('finalImageL',  0),
                          Entry('finalImageR',  0),
                          Entry('finalImageL', +1),
                          Entry('finalImageR', +1),
                          Entry('forwardFlowL',  0),
                          Entry('forwardFlowR',  0),
                          Entry('dispL',  0),
                          Entry('dispL', +1),
                          Entry('forwardDispChangeL', 0))),)

  nout = NumberOfEntries(samples)
  if 'output_index' in kwargs:
      nout += 1
      del kwargs['output_index']

  blobs = Layers.BinaryData(net,
                           nout=nout,
                           include=(Proto.NetStateRule(phase=kwargs['phase']),),
                           data_param=DataParams(samples, **kwargs))

  if kwargs['return_struct']: return _dataStructFromSceneFlow(blobs)
  else: return blobs


def BinaryData_SceneFlow_Single_Reduced(net, **kwargs):
  '''
  @brief Setup network inputs for scene flow
  @returns A list of single-blob network INPUT and LABEL
  '''
  samples = []

  if kwargs['rendertype'] == 'CLEAN' or kwargs['rendertype'] == 'BOTH':
      samples += (Sample((Entry('cleanImageL',  0),
                          Entry('cleanImageR',  0),
                          Entry('cleanImageL', +1),
                          Entry('cleanImageR', +1),
                          Entry('forwardFlowL',  0),
                          Entry('dispL',  0),
                          Entry('forwardDispChangeL', 0))),)

  if kwargs['rendertype'] == 'FINAL' or kwargs['rendertype'] == 'BOTH':
      samples += (Sample((Entry('finalImageL',  0),
                          Entry('finalImageR',  0),
                          Entry('finalImageL', +1),
                          Entry('finalImageR', +1),
                          Entry('forwardFlowL',  0),
                          Entry('dispL',  0),
                          Entry('forwardDispChangeL', 0))),)

  nout = NumberOfEntries(samples)
  if 'output_index' in kwargs:
      nout += 1
      del kwargs['output_index']

  blobs = Layers.BinaryData(net,
                           nout=nout,
                           include=(Proto.NetStateRule(phase=kwargs['phase']),),
                           data_param=DataParams(samples, **kwargs))

  if kwargs['return_struct']: return _dataStructFromSceneFlow2(blobs)
  else: return blobs


def BinaryData_Disparity(net, **kwargs):
  '''
  @brief Setup network inputs for disparity
  @returns A two-element list of single-blob network INPUT and LABEL
  '''

  samples = []

  if kwargs['rendertype'] == 'CLEAN' or kwargs['rendertype'] == 'BOTH':
      samples += (Sample((Entry('cleanImageL',  0),
                          Entry('cleanImageR',  0),
                          Entry('dispL',  0))),
                  Sample((Entry('cleanImageR',  0),
                          Entry('cleanImageL',  0),
                          Entry('dispR',  0))))

  if kwargs['rendertype'] == 'FINAL' or kwargs['rendertype'] == 'BOTH':
      samples += (Sample((Entry('finalImageL',  0),
                          Entry('finalImageR',  0),
                          Entry('dispL',  0))),
                  Sample((Entry('finalImageR',  0),
                          Entry('finalImageL',  0),
                          Entry('dispR',  0))))

  nout = NumberOfEntries(samples)
  if 'output_index' in kwargs:
      nout += 1
      del kwargs['output_index']

  return Layers.BinaryData(net,
                           nout=nout,
                           include=(Proto.NetStateRule(phase=kwargs['phase']),),
                           data_param=DataParams(samples, **kwargs))


def BinaryData_Disparity_Single(net, **kwargs):
  '''
  @brief Setup network inputs for disparity
  @returns A two-element list of single-blob network INPUT and LABEL
  '''

  samples = []

  if kwargs['rendertype'] == 'CLEAN' or kwargs['rendertype'] == 'BOTH':
      samples += (Sample((Entry('cleanImageL',  0),
                          Entry('cleanImageR',  0),
                          Entry('dispL',  0))),)

  if kwargs['rendertype'] == 'FINAL' or kwargs['rendertype'] == 'BOTH':
      samples += (Sample((Entry('finalImageL',  0),
                          Entry('finalImageR',  0),
                          Entry('dispL',  0))),)

  nout = NumberOfEntries(samples)
  if 'output_index' in kwargs:
      nout += 1
      del kwargs['output_index']

  return Layers.BinaryData(net,
                           nout=nout,
                           include=(Proto.NetStateRule(phase=kwargs['phase']),),
                           data_param=DataParams(samples, **kwargs))


def BinaryData(net, setting, **kwargs):
  '''
  @brief Setup network inputs by instantiating a BinaryDataLayer
  @returns A list of single-blob network INPUT and LABEL
  '''
  BinaryDataConstructors = {
    'OPTICAL_FLOW'               : BinaryData_OpticalFlow,
    'OPTICAL_FLOW_SINGLE'        : BinaryData_OpticalFlow_Single,
    'SCENE_FLOW'                 : BinaryData_SceneFlow,
    'SCENE_FLOW_SINGLE'          : BinaryData_SceneFlow_Single,
    'SCENE_FLOW_SINGLE_REDUCED'  : BinaryData_SceneFlow_Single_Reduced,
    'DISPARITY'                  : BinaryData_Disparity,
    'DISPARITY_SINGLE'           : BinaryData_Disparity_Single,
  }

  if 'collection_list' not in kwargs: 
    raise Exception('BinaryData requires parameter collectionList')

  def default(arg, val):
    if not arg in kwargs:
      kwargs[arg] = val
  
  default('rendertype',        'CLEAN')
  default('phase',             'TEST')
  default('batch_size',        1)
  default('verbose',           True)
  default('rand_permute',      True)
  default('rand_permute_seed', 77)
  default('collection_list_dir', COLLECTIONLIST_DIR)
  default('disk_reader_threads', 4)

  if setting in ('SCENE_FLOW', 'SCENE_FLOW_SINGLE', 'SCENE_FLOW_SINGLE_REDUCED'):
    default('return_struct', True)
  else:
    default('return_struct', False)

  if kwargs['phase'] == 'TEST': kwargs['phase'] = Proto.TEST
  if kwargs['phase'] == 'TRAIN': kwargs['phase'] = Proto.TRAIN

  if not os.path.isfile(os.path.join(kwargs['collection_list_dir'],
                                     kwargs['collection_list'])): 
    raise Exception('BinaryData: collection_list %s does not exist' \
                      %(os.path.join(kwargs['collection_list_dir'],
                                     kwargs['collection_list'])))

  if 'bin_db_dir' not in kwargs:
      if 'ssd_storage' in kwargs and kwargs['ssd_storage'] == True:
          kwargs['bin_db_dir'] = SSD_BIN_DB_DIR
      else:
          kwargs['bin_db_dir'] = BIN_DB_DIR

  return BinaryDataConstructors[setting](net, **kwargs)
