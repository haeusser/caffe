#!/usr/bin/python

from CaffeAdapter import  *
from caffe.proto import caffe_pb2 as Proto
from pymill import Toolbox as tb
import os


BIN_DB_DIR   = '/scratch/global/hackathon/data/4_bin-db'
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
  if isinstance(samples, tuple):
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
            'backend'   : Params.Data.BINARYDB,
            'batch_size': batch_size,
            'sample'    : samples,
            'verbose'   : verbose,
            'rand_permute'     : rand_permute}

  if rand_permute_seed is not None:
      params['rand_permute_seed'] = rand_permute_seed

  return params


def BinaryData_OpticalFlow(net, **kwargs):
  '''
  @brief Setup network inputs for optical flow
  @returns A list of single-blob network INPUT and LABEL
  '''
  samples = (Sample((Entry('finalImageL',  0),
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

  return Layers.BinaryData(net,
                           nout=NumberOfEntries(samples),
                           include=(Proto.NetStateRule(phase=kwargs['phase']),),
                           data_param=DataParams(samples, **kwargs))


def BinaryData_SceneFlow(net, **kwargs):
  '''
  @brief Setup network inputs for scene flow
  @returns A list of single-blob network INPUT and LABEL
  '''
  samples = (Sample((Entry('finalImageL',  0),
                     Entry('finalImageR',  0),
                     Entry('finalImageL', +1),
                     Entry('finalImageR', +1),
                     Entry('forwardFlowL',  0),
                     Entry('forwardFlowR',  0),
                     Entry('dispL',  0),
                     Entry('dispL', +1),
                     Entry('forwardDispChangeL', 0),
                     Entry('forwardDispChangeR', 0))),
             Sample((Entry('finalImageL',  0),
                     Entry('finalImageR',  0),
                     Entry('finalImageL', -1),
                     Entry('finalImageR', -1),
                     Entry('backwardFlowL',  0),
                     Entry('backwardFlowR',  0),
                     Entry('dispL',  0),
                     Entry('dispL', -1),
                     Entry('backwardDispChangeL', 0),
                     Entry('backwardDispChangeR', 0))))

  return Layers.BinaryData(net,
                           nout=NumberOfEntries(samples),
                           include=(Proto.NetStateRule(phase=kwargs['phase']),),
                           data_param=DataParams(samples, **kwargs))

#def instantiate(net):
  #net.input, net.gt = BinaryData(
      #setting='OPTICAL_FLOW',
      #collectionList='cliplist_040915_train.txt',
      #batch_size=4,
      #verbose=True,
      #rand_permute=True,
      #rand_permute_seed=77
    #)



def BinaryData_Disparity(net, **kwargs):
  '''
  @brief Setup network inputs for disparity
  @returns A two-element list of single-blob network INPUT and LABEL
  '''
  samples = (Sample((Entry('finalImageL',  0),
                     Entry('finalImageR',  0),
                     Entry('dispL',  0))),
             Sample((Entry('finalImageR',  0),
                     Entry('finalImageL',  0),
                     Entry('dispR',  0))))

  return Layers.BinaryData(net,
                           nout=NumberOfEntries(samples),
                           include=(Proto.NetStateRule(phase=kwargs['phase']),),
                           data_param=DataParams(samples, **kwargs))


def BinaryData(net, setting, **kwargs):
  '''
  @brief Setup network inputs by instantiating a BinaryDataLayer
  @returns A list of single-blob network INPUT and LABEL
  '''
  BinaryDataConstructors = {
    'OPTICAL_FLOW': BinaryData_OpticalFlow,
    'SCENE_FLOW'  : BinaryData_SceneFlow,
    'DISPARITY'   : BinaryData_Disparity,
  }

  if 'collection_list' not in kwargs: 
    raise Exception('BinaryData requires parameter collectionList')


  def default(arg, val):
    if not arg in kwargs:
      kwargs[arg] = val
  
  default('phase',             Proto.TRAIN)
  default('batch_size',        1)
  default('verbose',           True)
  default('rand_permute',      True)
  default('rand_permute_seed', 77)
  default('bin_db_dir',          BIN_DB_DIR)
  default('collection_list_dir', COLLECTIONLIST_DIR)
    
  if not os.path.isfile(os.path.join(kwargs['collection_list_dir'],
                                     kwargs['collection_list'])): 
    raise Exception('BinaryData: collection_list %s does not exist' \
                      %(kwargs['collection_list']))

  #
  # TODO: if kwargs['phase'] is not set do not add include: { phase: .. }
  # TODO: if kwargs['phase'] is set to 'TRAIN' do not add include: { phase: TRAIN }
  # TODO: if kwargs['phase'] is set to 'TEST' do not add include: { phase: TEST }
  #

  return BinaryDataConstructors[setting](net, **kwargs)
