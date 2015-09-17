#!/usr/bin/python

from CaffeAdapter import  *
from caffe.proto import caffe_pb2 as Proto
from pymill import Toolbox as tb
import os


def PhilData_OpticalFlow(net, **kwargs):
  '''
  @brief Setup PhilDataLayer for optical flow
  @returns A list of layer output blobs
  '''
  data_param = {'source'     : kwargs['source'],
                'backend'    : Params.Data.LMDB,
                'batch_size' : kwargs[''],
                'encoding  ' : (Proto.DataParameter.UINT8,
                                Proto.DataParameter.UINT8,
                                Proto.DataParameter.UINT16FLOW,
                                Proto.DataParameter.BOOL1),
                'slice_point': (3, 6, 8),
                'verbose'    : kwargs['verbose'],
                'preselection_file'  : kwargs['preselection_file'],
                'preselection_label' : kwargs['preselection_label'],
                'rand_permute'       : kwargs['rand_permute'],
                'rand_permute_seed'  : kwargs['rand_permute_seed']}
  
  ## Always returns (img_from, img_to, flow, occlusion)
  return Layers.PhilData(net, nout=4,
                         include=(Proto.NetStateRule(phase=kwargs['phase']),),
                         data_param=data_param)


def PhilData(net, **kwargs):
  '''
  @brief Setup network inputs by instantiating a PhilDataLayer
  @returns A list of single-blob network INPUT and LABEL
  '''

  if 'source' not in kwargs: 
    raise Exception('PhilData requires parameter >source<')
  if 'preselection_file' not in kwargs: 
    raise Exception('PhilData requires parameter >preselection_file<')
  
  if not os.path.isfile(kwargs['source']): 
    raise Exception('PhilData: >source< %s does not exist'
                    %(kwargs['source']))
  if not os.path.isfile(kwargs['preselection_file']): 
    raise Exception('PhilData: >preselection_file< %s does not exist'
                    %(kwargs['preselection_file']))


  def default(arg, val):
    if not arg in kwargs:
      kwargs[arg] = val
  
  default('phase',              Proto.TRAIN)
  default('batch_size',         1)
  default('verbose',            True)
  default('rand_permute',       True)
  default('rand_permute_seed',  77)
  default('preselection_label', 1)
    

  return MakePhilData(net, **kwargs)
