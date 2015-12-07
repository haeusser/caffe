// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

namespace caffe {

static __device__ __forceinline__ float gauss(float x, float sigma)
{
    float exponent = x / sigma;
    return exp( -0.5 * exponent *exponent);
}

template <typename Dtype>
__global__ void GaussKernelX(
        const int nthreads,
        const Dtype* in_ptr,
        const int width,
        const int height,
        const int channelsize,
        const float sigma,
        Dtype* out_ptr)
{
    CUDA_KERNEL_LOOP(index, nthreads)
    {
        int c = index / channelsize;
        int x = (index % channelsize) % width;
        int y = (index % channelsize) / width;

        Dtype sum=0;
        Dtype wsum=0;

        int rx = 3*sigma;

        for(int xf=x-rx; xf<=x+rx; xf++)
        {
            if(xf<0) continue;
            if(xf>=width) continue;

            int dx = abs(xf - x);
            float w = gauss(dx, sigma);
            sum += w * in_ptr[c*channelsize + y*width+xf];
            wsum += w;
        }

        out_ptr[index] = (wsum == 0) ? 0 : (sum / wsum);
    }
}

template <typename Dtype>
__global__ void GaussKernelY(
        const int nthreads,
        const Dtype* in_ptr,
        const int width,
        const int height,
        const int channelsize,
        const float sigma,
        Dtype* out_ptr)
{
    CUDA_KERNEL_LOOP(index, nthreads)
    {
        int c = index / channelsize;
        int x = (index % channelsize) % width;
        int y = (index % channelsize) / width;

        Dtype sum=0;
        Dtype wsum=0;

        int ry = 3*sigma;

        for(int yf=y-ry; yf<=y+ry; yf++)
        {
            if(yf<0) continue;
            if(yf>=height) continue;

            int dy = abs(yf - y);
            float w = gauss(dy, sigma);
            sum += w * in_ptr[c*channelsize + yf*width+x];
            wsum += w;
        }

        out_ptr[index] = (wsum == 0) ? 0 : (sum / wsum);
    }
}

template <typename Dtype>
void BlurLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  Dtype* top_data = top[0]->mutable_gpu_data(); // dest
  
  Dtype* bottom_data = bottom[0]->mutable_gpu_data(); // source
  int bottomnum = (bottom)[0]->num();
  int bottomchannels = (bottom)[0]->channels();
  int bottomwidth = (bottom)[0]->width();
  int bottomheight = (bottom)[0]->height();
  int bottomcount = (bottom)[0]->count();
   
  float sigma = this->layer_param().blur_param().sigma();

  int botsize = bottomwidth*bottomheight*bottomchannels*bottomnum;
  int botchannelsize = bottomwidth*bottomheight;

  GaussKernelX<Dtype><<<CAFFE_GET_BLOCKS(botsize), CAFFE_CUDA_NUM_THREADS>>>(
      botsize,
      (Dtype*)bottom_data,
      bottomwidth,
      bottomheight,
      botchannelsize,
      sigma,
      intermediate_.mutable_gpu_data());

  GaussKernelY<Dtype><<<CAFFE_GET_BLOCKS(botsize), CAFFE_CUDA_NUM_THREADS>>>(
      botsize,
      intermediate_.gpu_data(),
      bottomwidth,
      bottomheight,
      botchannelsize,
      sigma,
      (Dtype*)top_data);
}

template <typename Dtype>
void BlurLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  LOG(FATAL) << "BlurLayer cannot do backward.";
}

INSTANTIATE_LAYER_GPU_FUNCS(BlurLayer);

}  // namespace caffe



