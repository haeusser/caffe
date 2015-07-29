#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MaxUnpoolBackward(const int nthreads,
    const Dtype* const top_diff, const int num, const int channels,
    const int unpooled_height, const int unpooled_width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    Dtype* bottom_diff, const Dtype* const bottom_mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int i = bottom_mask[index];
    bottom_diff[index] += top_diff[i];
  }
}

template <typename Dtype>
void UnpoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  const Dtype* bottom_mask = bottom[1]->gpu_data();
  const int num = top[0]->num();
 
  // NOLINT_NEXT_LINE(whitespace/operators)
  MaxUnpoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff,num, channels_,
      unpooled_height_, unpooled_width_, pooled_height_, pooled_width_, kernel_h_,
      kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, bottom_diff,
      bottom_mask);
  
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void MaxUnpoolForward(const int nthreads, const Dtype* const bottom_data,
    const Dtype* const bottom_mask, const int num,
    const int channels, const int unpooled_height, const int unpooled_width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
 
    // find out the local index
    // find out the local offset
    const int w = index % unpooled_width;
    const int h = (index / unpooled_width) % unpooled_height;
    const int c = (index / unpooled_width / unpooled_height) % channels;
    const int n = index / unpooled_width / unpooled_height / channels;
    const int phstart =
         (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const int phend = min((h + pad_h) / stride_h + 1, pooled_height);
    const int pwstart =
         (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const int pwend = min((w + pad_w) / stride_w + 1, pooled_width);

    const int offset = (n * channels + c) * pooled_height * pooled_width;
    const Dtype* const bottom_data_slice = bottom_data + offset;
    
    const Dtype* const mask_slice = bottom_mask + offset;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
	if (mask_slice[ph * pooled_width + pw] == h * unpooled_width + w) {
	  top_data[index] = bottom_data_slice[ph * pooled_width + pw];
	}
      }
    }
  }
}

template <typename Dtype>
void UnpoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_mask = bottom[1]->gpu_data();  
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = top[0]->count();
  caffe_gpu_set(top[0]->count(), Dtype(0.), top_data);

  // NOLINT_NEXT_LINE(whitespace/operators)
  MaxUnpoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, bottom_mask, bottom[0]->num(), channels_,
      unpooled_height_, unpooled_width_, pooled_height_, pooled_width_,
      kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
      top_data);

  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(UnpoolingLayer);


}  // namespace caffe
