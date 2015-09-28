#!/usr/bin/python

from BinaryData import BinaryData
from PhilData import PhilData

def FlyingChairs(net, **kwargs):
  if 'source' in kwargs: raise Exception('Chairs cannot take >source<')
  if 'preselection_file' in kwargs: raise Exception('Chairs cannot take >preselection_file<')
  if 'preselection_label' in kwargs: raise Exception('Chairs cannot take >preselection_label<')

  kwargs['source'] = '/misc/lmbraid17/sceneflownet/common/data/3_other-ds/FlyingChairs/FlyingChairs-FlowNet-lmdb'

  if 'subset' in kwargs:
      if kwargs['subset'] == 'TRAIN':
          kwargs['preselection_file'] = '/misc/lmbraid17/sceneflownet/common/data/3_other-ds/FlyingChairs/FlyingChairs-FlowNet-preselection.txt'
          kwargs['preselection_label'] = 1
      elif kwargs['subset'] == 'TEST':
          kwargs['preselection_file'] = '/misc/lmbraid17/sceneflownet/common/data/3_other-ds/FlyingChairs/FlyingChairs-FlowNet-preselection.txt'
          kwargs['preselection_label'] = 2
      elif kwargs['subset'] != 'ALL':
          raise Exception('Subset must be one of TRAIN, TEST, ALL')

  return PhilData(net, **kwargs)

def Sintel(net, **kwargs):
  if 'source' in kwargs: raise Exception('Sintel cannot take >source<')
  if 'preselection_file' in kwargs: raise Exception('Sintel cannot take >preselection_file<')
  if 'preselection_label' in kwargs: raise Exception('Sintel cannot take >preselection_label<')
  if 'collection_list' in kwargs: raise Exception('Sintel cannot take >collection_list<')

  kwargs['collection_list'] = '/misc/lmbraid17/sceneflownet/common/data/4_bin-db/collection_lists/sintel_train.txt'

  return BinaryData(net, **kwargs)

