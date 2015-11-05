#!/usr/bin/python

import os
import re
import numpy as np
import uuid
from scipy import misc
import numpy as np
from PIL import Image
import sys


def readPFM(file):
    file = open(file, 'rb')

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

def writePFM(file, image, scale=1):
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')
      
    image = np.flipud(image)  

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n')
    file.write('%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n' % scale)

    image.tofile(file)

def readFlow(name):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        return readPFM(name)[0][:,:,0:2]

    f = open(name, 'rb')

    header = f.read(4)
    if header != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1)
    height = np.fromfile(f, np.int32, 1)

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

    return flow.astype(np.float32)

def readImage(name):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        return readPFM(name)[0][:,:,0:3]

    return misc.imread(name)

def writeFlow(name, flow):
    f = open(name, 'wb')
    f.write('PIEH')
    np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
    flow = flow.astype(np.float32)
    flow.tofile(f)

def readFloat(name):
    f = open(name, 'rb')

    if f.readline() != 'float\n':
        raise Exception('float file %s did not contain <float> keyword' % name)

    dim = int(f.readline())

    dims = []
    count = 1
    for i in range(0, dim):
        d = int(f.readline())
        dims.append(d)
        count *= d

    dims = list(reversed(dims))

    data = np.fromfile(f, np.float32, count).reshape(dims)
    if dim > 2:
        data = np.transpose(data, (2, 1, 0))
        data = np.transpose(data, (1, 0, 2))

    return data

def writeFloat(name, data):
    f = open(name, 'wb')

    dim=len(data.shape)
    if dim>3:
        raise Exception('bad float file dimension: %d' % dim)

    f.write('float\n')
    f.write('%d\n' % dim)

    if dim == 1:
        f.write('%d\n' % data.shape[0])
    else:
        f.write('%d\n' % data.shape[1])
        f.write('%d\n' % data.shape[0])
        for i in range(2, dim):
            f.write('%d\n' % data.shape[i])

    data = data.astype(np.float32)
    if dim==2:
        data.tofile(f)

    else:
        np.transpose(data, (2, 0, 1)).tofile(f)

def writeDisparity(filename,disparity,bitdepth=16):
    """ Write disparity to file.

    bitdepth can be either 16 (default) or 32.

    The maximum disparity is 1024, since the image width in Sintel
    is 1024.
    """
    d = disparity.copy()

    # Clip disparity.
    d[d>1024] = 1024
    d[d<0] = 0

    d_r = (d / 4.0).astype('uint8')
    d_g = ((d * (2.0**6)) % 256).astype('uint8')

    out = np.zeros((d.shape[0],d.shape[1],3),dtype='uint8')
    out[:,:,0] = d_r
    out[:,:,1] = d_g

    if bitdepth > 16:
        d_b = (d * (2**14) % 256).astype('uint8')
        out[:,:,2] = d_b

    Image.fromarray(out,'RGB').save(filename,'PNG')


def readDisparity(filename):
    if filename.endswith('.pfm') or filename.endswith('.PFM'):
        disp, scale = readPFM(filename)
        return disp

    """ Return disparity read from filename. """

    f_in = np.array(Image.open(filename))

    ## Case 1: 8-bit RGB
    if len(f_in.shape) == 3:
      d_r = f_in[:,:,0].astype('float64')
      d_g = f_in[:,:,1].astype('float64')
      d_b = f_in[:,:,2].astype('float64')
      depth = d_r * 4 + d_g / (2**6) + d_b / (2**14)
    ## Case 2: 16-bit grayscale PNG
    elif len(f_in.shape) == 2:
      depth = f_in[:,:].astype('float64') / 256.
      ## Set invalid pixels (disparity 0) to -1
      depth[depth==0] = np.nan
    else:
      raise Exception('Invalid image format')

    return depth


def readList(filename):
    list = []
    for line in open(filename).readlines():
        if line.strip() != '':
            list.append(line.split()[0])

    return list

def readTupleList(filename):
    list = []
    for line in open(filename).readlines():
        if line.strip() != '':
            list.append(line.split())

    return list

def openLMDB(path, create=False):
    import lmdb
    if create and (not os.path.isdir(path)):
        os.makedirs(path)
    return lmdb.open(path, map_size=1024 * 1024 * 1024 * 1024, max_readers=1022)

def avprobe(file):
    info = run('avprobe %s 2>&1' % file)

    data = {}

    lines = info.split('\n')
    tag = ''
    subtag = ''
    subsubtag = ''
    for l in lines:
        match = re.match('^Input #([0-9]+),', l)
        if match:
            tag = match.group(0).replace(',','')
            subtag = ''
            subsubtag = ''
            data[tag] = {}
            data[tag]['='] = l.replace(match.group(0),'').strip()

        match = re.match('^[ ]{2}([a-zA-Z0-9]+):', l)
        if match:
            subtag = match.group(1)
            subsubtag = ''
            data[tag][subtag] = {}
            data[tag][subtag]['='] = l.replace(match.group(0),'').strip()

        match = re.match('^[ ]{4}([a-zA-Z0-9]+.*?):', l)
        if match:
            subsubtag = match.group(1).strip()
            data[tag][subtag][subsubtag] = {}
            data[tag][subtag][subsubtag]['='] = l.replace(match.group(0), '').strip()

    return data

def avinfo(file):
    data = avprobe(file)

    info = {}
    if not 'Input #0' in data:
        return info

    branch = data['Input #0']
    info['format'] = branch['Metadata']['major_brand']['=']

    branch = branch['Duration']
    duration = branch['=']

    match = re.match('.*?([0-9]{2}):([0-9]{2}):([0-9]{2})\.[0-9]{2}', duration)
    if match:
        info['duration'] = int(match.group(1))*60*60 + int(match.group(2))*60 + int(match.group(3))

    for tag in branch:
        if tag == '=': continue
        value = branch[tag]['=']
        if value.startswith('Video'):
            match = re.search('([0-9]+)x([0-9]+)', value)
            if match:
                info['width'] = int(match.group(1))
                info['height'] = int(match.group(2))

            match = re.search('([0-9\.]+) fps', value)
            if match:
                info['fps'] = float(match.group(1))

            match = re.match('Video: (.*?),', value)
            if match:
                info['codec'] = match.group(1)

            match = re.search('([0-9]+) kb/s', value)
            if match:
                info['avgbitrate'] = int(match.group(1))

    import cv2
    cap = cv2.VideoCapture(file)
    info['frames'] = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    info['size'] = (info['duration'] * info['avgbitrate']) / 8.0 / 1024.0

    info['compression'] = info['size'] * 1024 * 1024 / (info['width'] * info['height'] * 3 * info['frames'])
    return info

def tempFilename(ext=''):
    os.system('mkdir -p /tmp/ilge/python')
    return '/tmp/ilge/python/%s%s' % (uuid.uuid4(), ext)

def recv_size(socket, size):
    buf = bytearray()
    while(len(buf)<size):
        diff = size - len(buf)
        buf += bytearray(socket.recv(diff))
    return buf
