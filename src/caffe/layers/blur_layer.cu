// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

namespace caffe {

static __device__ __forceinline__ float gauss(float x, float sigma, float prefactor)
{
    float exponent = x / sigma;
    exponent = exponent * exponent;
    return prefactor*exp(-0.5*exponent);
}

#define FILTER_GAUSSIAN 0

template <typename Dtype>
__global__ void BlurKernel(
        const int nthreads,
        const Dtype* in_ptr,
        const int width,
        const int height,
        const int channelsize,
        const int filter_type,
        const float sigma,
        const float prefactor,
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
        int ry = 3*sigma;

        for(int yf=y-ry; yf<=y+ry; yf++)
            for(int xf=x-rx; xf<=x+rx; xf++)
            {
                if(yf<0 || xf<0) continue;
                if(yf>=height || xf>=width) continue;

                int dx = xf - x;
                int dy = yf - y;
                float r = sqrt(double(dx*dx + dy*dy));
                float w = gauss(r, sigma, prefactor);
                sum += w * in_ptr[c*channelsize + yf*width+xf];
                wsum += w;
            }

        out_ptr[index] = (!wsum) ? 0 : (sum / wsum);
    }
}

template <typename Dtype>
__global__ void GaussKernelX(
        const int nthreads,
        const Dtype* in_ptr,
        const int width,
        const int height,
        const int channelsize,
        const float sigma,
        const float prefactor,
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
            float w = gauss(dx, sigma, prefactor);
            sum += w * in_ptr[c*channelsize + y*width+xf];
            wsum += w;
        }

        out_ptr[index] = (!wsum) ? 0 : (sum / wsum);
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
        const float prefactor,
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
            float w = gauss(dy, sigma, prefactor);
            sum += w * in_ptr[c*channelsize + yf*width+x];
            wsum += w;
        }

        out_ptr[index] = (!wsum) ? 0 : (sum / wsum);
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

  float prefactor = 1.0/(sigma*sqrt(2*M_PI));

  GaussKernelX<Dtype><<<CAFFE_GET_BLOCKS(botsize), CAFFE_CUDA_NUM_THREADS>>>(
      botsize,
      (Dtype*)bottom_data,
      bottomwidth,
      bottomheight,
      botchannelsize,
      sigma,
      prefactor,
      (Dtype*)top_data);

//  GaussKernelY<Dtype><<<CAFFE_GET_BLOCKS(botsize), CAFFE_CUDA_NUM_THREADS>>>(
//      botsize,
//      (Dtype*)top_data,
//      bottomwidth,
//      bottomheight,
//      botchannelsize,
//      sigma,
//      prefactor,
//      (Dtype*)top_data);
}

template <typename Dtype>
void BlurLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  LOG(FATAL) << "ResampleLayer cannot do backward.";
}

INSTANTIATE_LAYER_GPU_FUNCS(BlurLayer);

}  // namespace caffe






//      cv::gpu::GpuMat input(bottomheight, bottomwidth, CV_32FC3);
//      float* input_ptr=(float*)input.data;
//      int input_stride=input.step/4;
//      BlobToOpenCV<Dtype><<<CAFFE_GET_BLOCKS(bottomwidth*bottomheight), CAFFE_CUDA_NUM_THREADS>>>(
//              bottomwidth*bottomheight,
//              (Dtype*)bottom_data,
//              bottomwidth,
//              bottomheight,
//              input_stride,
//              (Dtype*)input_ptr);


//      cv::gpu::GpuMat output;
//      cv::Size output_size;
//      output_size.width = topwidth;
//      output_size.height = topheight;
//      cv::gpu::resize(input,output,output_size,0,0,interpolation,cv::gpu::Stream::Null(),false);

//      float* output_ptr=(float*)output.data;
//      int output_stride=output.step/4;
//      OpenCVToBlob<Dtype><<<CAFFE_GET_BLOCKS(topwidth*topheight), CAFFE_CUDA_NUM_THREADS>>>(
//              topwidth*topheight,
//              (Dtype*)output_ptr,
//              topwidth,
//              topheight,
//              output_stride,
//              (Dtype*)top_data);

//      top_data += topsize;
//      bottom_data += botsize;

//template <typename Dtype>
//__global__ void BlobToOpenCV(
//        const int nthreads,
//        const Dtype* blob_ptr,
//        const int width,
//        const int height,
//        const int stride,
//        Dtype* mat_ptr)
//{
//    CUDA_KERNEL_LOOP(index, nthreads)
//    {
//        int x=index % width;
//        int y=index / width;


//        for(int c=0; c<3; c++)
//            mat_ptr[y*stride+x*3+c]=blob_ptr[((c*height)+y)*width+x];
//    }
//}

//template <typename Dtype>
//__global__ void OpenCVToBlob(
//        const int nthreads,
//        const Dtype* mat_ptr,
//        const int width,
//        const int height,
//        const int stride,
//        Dtype* blob_ptr)
//{
//    CUDA_KERNEL_LOOP(index, nthreads)
//    {
//        int x=index % width;
//        int y=index / width;


//        for(int c=0; c<3; c++)
//            blob_ptr[((c*height)+y)*width+x]=mat_ptr[y*stride+x*3+c];
//    }
//}
