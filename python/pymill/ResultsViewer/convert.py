#!/usr/bin/env python3.4

#######################################################################
## Nikolaus Mayer, 2015                                              ##
#######################################################################

from math import sqrt
import numpy as np
import os
import sys
import re
import platform
import gridview_c_interface as c_interface



def readPFM(file):
  '''Read .pfm files'''
  with open(file, 'rb') as file:
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == 'PF':
      color = True
    elif header == 'Pf':
      color = False
    else:
      raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
      width, height = map(int, dim_match.groups())
    else:
      raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
      endian = '<'
      scale = -scale
    else:
      endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


def readFlow(name):
  '''Read optical flow images from .flo or .pfm files'''
  if name.endswith('.pfm') or name.endswith('.PFM'):
    return readPFM(name)[0][:,:,0:2]
  with open(name, 'rb') as f:
    header = f.read(4)
    if header != b'PIEH':
      raise Exception('Flow file (%s) header does not contain PIEH'\
                      %(name))
    width = np.fromfile(f, np.int32, 1)
    height = np.fromfile(f, np.int32, 1)
    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))
    return flow.astype(np.float32)


def readFloat(name):
  '''Read float file (binary blob with leading dimension info)'''
  with open(name, 'rb') as f:
    if f.readline() != b'float\n':
      raise Exception('float file %s did not contain <float> keyword' % name)

    dim = int(f.readline())
    dims = []
    count = 1
    for i in range(0, dim):
      d = int(f.readline())
      dims.append(d)
      count *= d

    ## Hacky hack. multichannel data -> use first channel
    if dim == 3:
      dim = 2
      dims = dims[:2]
      dims = dims[::-1]

    data = np.fromfile(f, np.float32, count).reshape(dims)
    if dim == 2:
      #data = np.transpose(data, (1, 0))
      data = np.transpose(data, (0, 1))
    else:
      raise Exception('dimensions (%s) not supported'%(','.join(dims)))
    """elif dim == 3:
      if dims[2] == 1:
        data = data[:,:,0]
        #data = np.transpose(data, (2, 1, 0))
        #data = np.transpose(data, (1, 0, 2))
        #data = np.transpose(data, (1, 0))
        #data = np.transpose(data, (0, 1))
      else:
        raise Exception('3d data not (yet) supported')
        #data = np.transpose(data, (2, 1, 0))
        #data = np.transpose(data, (1, 0, 2))"""

    return data


def writeFloat(name, data):
  with open(name, 'wb') as f:

    dim=len(data.shape)
    if dim>3:
      raise Exception('bad float file dimension: %d' % dim)

    def ascii(s):
      return bytes(s, 'ascii')

    f.write(ascii('float\n'))
    f.write(ascii('%d\n' % dim))

    if dim == 1:
      f.write(ascii('%d\n' % data.shape[0]))
    else:
      f.write(ascii('%d\n' % data.shape[1]))
      f.write(ascii('%d\n' % data.shape[0]))
      for i in range(2, dim):
        f.write(ascii('%d\n' % data.shape[i]))

    data = data.astype(np.float32)
    if dim==2:
      data.transpose().tofile(f)
    else:
      np.transpose(data, (2, 0, 1)).tofile(f)


def writeFlow(name, flow):
  with open(name, 'wb') as f:
    f.write(b'PIEH')
    np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
    flow = flow.astype(np.float32)
    flow.tofile(f)


def FlowDelta(a, b):
  '''Flow vector difference'''
  result = c_interface.FlowDelta(a, b)
  return result


def EPE(a, b):
  '''Pixelwise scalar end-point-error'''
  result = c_interface.PixelwiseEPE(a, b)
  return result
  


def files(dir, suffix, batch):
  if batch:
    ## Woops
    raise Exception('NOT IMPLEMENTED')
  else:
    template = os.path.join(dir, '%07d%s')
    i = 0
    while True:
      yield template%(i,suffix)
      i += 1


def make_diff_images(dir, template_a, template_b, template_c, batch):
  for (a,b,diff) in zip(files(dir, template_a, batch), 
                        files(dir, template_b, batch), 
                        files(dir, template_c, batch)):
    if not os.path.isfile(a) or not os.path.isfile(b):
      break
    print(diff)
    a_img    = readFlow(a)
    b_img    = readFlow(b)
    diff_img = FlowDelta(a_img, b_img)
    writeFlow(diff, diff_img)

  

def make_EPE_images(dir, template_a, template_b, template_c, batch):
  for (a,b,diff) in zip(files(dir, template_a, batch), 
                        files(dir, template_b, batch), 
                        files(dir, template_c, batch)):
    if not os.path.isfile(a) or not os.path.isfile(b):
      break
    print(diff)
    a_img    = readFlow(a)
    b_img    = readFlow(b)
    diff_img = EPE(a_img, b_img)
    writeFloat(diff, diff_img)



def main():

  if len(sys.argv) < 2:
    print('Usage: %s <data folder>'%(sys.argv[0]))
    return

  datafolder = sys.argv[1]

  def predicted_file():
    template = os.path.join(datafolder, '%07d-flow.flo')
    i = 0
    while True:
      yield template%(i)
      i += 1

  def gt_file():
    template = os.path.join(datafolder, '%07d-gt.flo')
    i = 0
    while True:
      yield template%(i)
      i += 1

  ''''def diff_file():
    template = os.path.join(datafolder, '%07d-diff.float')
    i = 0
    while True:
      yield template%(i)
      i += 1

  for (pre,gt,diff) in zip(predicted_file(), gt_file(), diff_file()):
    if not os.path.isfile(pre):
      break
    print(diff)
    pre_img  = readFlow(pre)
    gt_img   = readFlow(gt)
    diff_img = EPE(pre_img, gt_img)
    writeFloat(diff, diff_img)'''


  def diff_file():
    template = os.path.join(datafolder, '%07d-diff.flo')
    i = 0
    while True:
      yield template%(i)
      i += 1

  for (pre,gt,diff) in zip(predicted_file(), gt_file(), diff_file()):
    if not os.path.isfile(pre):
      break
    print(diff)
    pre_img  = readFlow(pre)
    gt_img   = readFlow(gt)
    diff_img = FlowDelta(pre_img, gt_img)
    writeFlow(diff, diff_img)


if __name__ == '__main__':
  main()


