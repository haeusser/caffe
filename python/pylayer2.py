import caffe
import numpy as np

class PythonLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 1:
            raise Exception("Need one input.")

    def reshape(self, bottom, top):
        print('Going to print')
        print('Ashapes bottom[0] (n,c,h,w) = %d,%d,%d,%d' % (bottom[0].num,bottom[0].channels,bottom[0].height,bottom[0].width))
        if top[0].width > 1:
            print('Ashapes top[0] (n,c,h,w) = %d,%d,%d,%d top[0].data = %s' % (top[0].num,top[0].channels,top[0].height,top[0].width,str(top[0].data)))
        #print('Ashapes bottom=%s top=%s' % (str(bottom[0].data.shape), str(top[0].data.shape)))
        top[0].reshape(bottom[0].num, bottom[0].channels, bottom[0].height, bottom[0].width)
        print('Bshapes bottom[0] (n,c,h,w) = %d,%d,%d,%d' % (bottom[0].num,bottom[0].channels,bottom[0].height,bottom[0].width))
        print('Bshapes top[0] (n,c,h,w) = %d,%d,%d,%d top[0].data = %s' % (top[0].num,top[0].channels,top[0].height,top[0].width,str(top[0].data)))
        #print('Bshapes bottom=%s top=%s' % (str(bottom[0].data.shape), str(top[0].data.shape)))

    def forward(self, bottom, top):
        top[0].data[...] = bottom[0].data

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            bottom[0].diff[...] = top[0].diff