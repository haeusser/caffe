#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void UnpoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  UnpoolingParameter unpool_param = this->layer_param_.unpooling_param();
}

template <typename Dtype>
void UnpoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input (pooled map) must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  CHECK_EQ(4, bottom[1]->num_axes()) << "Input (mask) must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  CHECK_EQ(bottom[0]->width(), bottom[1]->width()) 
      << "Pooled map and mask must have same width.";
  CHECK_EQ(bottom[0]->height(), bottom[1]->height()) 
      << "Pooled map and mask must have same height.";
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels()) 
      << "Pooled map and mask must have same number of channels.";
  
  channels_ = bottom[0]->channels();
  pooled_height_ = bottom[0]->height();
  pooled_width_ = bottom[0]->width();
  unpooled_height_ = static_cast<int>(ceil((pooled_height_ - 1) * stride_h_ - 2 * pad_h_ + kernel_h_));
  unpooled_width_ = static_cast<int>(ceil((pooled_width_ - 1) * stride_w_ - 2 * pad_w_ + kernel_w_));
  channels_ = bottom[0]->channels();
  pooled_count_ = bottom[0]->count();
      
  top[0]->Reshape(bottom[0]->num(), channels_, unpooled_height_,
      unpooled_width_);

}

template <typename Dtype>
void UnpoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  /*  THIS SHOULD PRODUCE THE SAME AS THE LOOP BELOW
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* mask = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(0), top_data);

  for (int n = 0; n < bottom[0]->num(); ++n) {
    for (int c = 0; c < channels_; ++c) {
      for (int idx = 0; idx < pooled_count_; ++idx) {
        top_data[mask[idx]] += bottom_data[idx];
      }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
        mask += top[0]->offset(0, 1);
    }
  }
  */

  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* mask = bottom[1]->cpu_data();  
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(0), top_data);
  
  for (int n = 0; n < bottom[0]->num(); ++n) {
    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          const int pooled_index = ph * pooled_width_ + pw;
          const int unpooled_index = mask[pooled_index];
          top_data[unpooled_index] = bottom_data[pooled_index];
        }
      }
      bottom_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
      mask += top[0]->offset(0, 1);

    }
  }
}

template <typename Dtype>
void UnpoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  // max pool over top_diff and output bottom_diff and mask_diff
  
  const Dtype* top_diff = top[0]->cpu_diff();  // const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_data_diff = bottom[0]->mutable_cpu_diff();  // Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* mask_diff = bottom[1]->mutable_cpu_diff();
  
  const int bottom_count = bottom[0]->count();
  caffe_set(bottom_count, Dtype(-FLT_MAX), bottom_data_diff);
  caffe_set(bottom_count, Dtype(-1), mask_diff);

  // The main loop
  for (int n = 0; n < top[0]->num(); ++n) {
    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          int hstart = ph * stride_h_ - pad_h_;
          int wstart = pw * stride_w_ - pad_w_;
          int hend = min(hstart + kernel_h_, unpooled_height_);
          int wend = min(wstart + kernel_w_, unpooled_width_);
          hstart = max(hstart, 0);
          wstart = max(wstart, 0);
          const int pool_index = ph * pooled_width_ + pw;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int index = h * unpooled_width_ + w;
              if (top_diff[index] > bottom_data_diff[pool_index]) {
                bottom_data_diff[pool_index] = top_diff[index];
                mask_diff[pool_index] = static_cast<Dtype>(index);
              }
            }
          }
        }
      }
      // compute offset
      top_diff += bottom[0]->offset(0, 1);
      bottom_data_diff += top[0]->offset(0, 1);
      mask_diff += top[0]->offset(0, 1);
      }
  }
}


#ifdef CPU_ONLY
STUB_GPU(UnpoolingLayer);
#endif

INSTANTIATE_CLASS(UnpoolingLayer);
REGISTER_LAYER_CLASS(Unpooling);

}  // namespace caffe
