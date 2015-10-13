// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/util/math_functions.hpp"

#include <iostream>
#include <fstream>

#define CUDART_NAN_F            __int_as_float(0x7fffffff)

namespace caffe {

  inline __device__ __host__ float clamp(float f, float a, float b)
{
    return fmaxf(a, fminf(f, b));
}


template <typename Dtype> 
__global__ void WarpData(const int nthreads, const int num, const int channels, const int height, const int width, const Dtype* src_data, const int src_count,
                                const int dest_height, const int dest_width, Dtype* dest_data, const Dtype* flow) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int x = (index % width); //w-pos
    int y = ((index / width) % height); //h-pos
    int cn = (index / width / height); // channel*num
    int n = cn / channels; //num
    
    // === Warping:
    
    float xpos = (float)(x) + flow[width*(height*(2*n+0) + y) + x];
    float ypos = (float)(y) + flow[width*(height*(2*n+1) + y) + x];
    
    if (xpos > 0.f && xpos <= width-1.01f && ypos > 0.f && ypos <= height-1.01f) {      
      // Get interpolated sample
      float tlx = floor(xpos);
      float tly = floor(ypos);
      int srcIdxOff = width*(height*cn + tly) + tlx;
      
      float sampleTL = src_data[srcIdxOff];
      float sampleTR = src_data[min(srcIdxOff+1,src_count)];
      float sampleBL = src_data[min(srcIdxOff+width,src_count)];
      float sampleBR = src_data[min(srcIdxOff+1+width,src_count)];
      
      float xdist = xpos - tlx;
      float ydist = ypos - tly;
      
      float sample = (1-xdist)*(1-ydist)*sampleTL
                  + (  xdist)*(  ydist)*sampleBR
                  + (1-xdist)*(  ydist)*sampleBL
                  + (  xdist)*(1-ydist)*sampleTR;
      
      dest_data[index] = sample;
    }
  }
}


template <typename Dtype>
void ApplyFlowLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  Dtype* top_data = top[0]->mutable_gpu_data(); // dest
  int topwidth = top[0]->width();
  int topheight = top[0]->height();
  int topchannels = top[0]->channels();
  int topcount = top[0]->count();
  
  const Dtype* bottom_data = bottom[0]->gpu_data(); // source image
  const Dtype* flow_data = bottom[1]->gpu_data(); // source flow
  
  int bottomchannels = (bottom)[0]->channels();
  int bottomwidth = (bottom)[0]->width();
  int bottomheight = (bottom)[0]->height();
  int bottomcount = (bottom)[0]->count();
  
  int num = (bottom)[0]->num();
  
  CHECK_EQ((bottom)[0]->num(), top[0]->num());
  
  caffe_gpu_set(top[0]-> count(), Dtype(0), top_data);
 
  WarpData<Dtype><<<CAFFE_GET_BLOCKS(topcount), CAFFE_CUDA_NUM_THREADS>>>(
      topcount,
      num, bottomchannels, bottomheight, bottomwidth, bottom_data, bottomcount,
      topheight, topwidth, top_data, flow_data);

  CUDA_POST_KERNEL_CHECK;
}



INSTANTIATE_LAYER_GPU_FUNCS(ApplyFlowLayer);

}  // namespace caffe
