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
  CHECK(!unpool_param.has_kernel_size() !=
    !(unpool_param.has_kernel_h() && unpool_param.has_kernel_w()))
    << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
  CHECK(unpool_param.has_kernel_size() ||
    (unpool_param.has_kernel_h() && unpool_param.has_kernel_w()))
    << "For non-square filters both kernel_h and kernel_w are required.";
  CHECK((!unpool_param.has_pad() && unpool_param.has_pad_h()
      && unpool_param.has_pad_w())
      || (!unpool_param.has_pad_h() && !unpool_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!unpool_param.has_stride() && unpool_param.has_stride_h()
      && unpool_param.has_stride_w())
      || (!unpool_param.has_stride_h() && !unpool_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  if (unpool_param.has_kernel_size()) {
    kernel_h_ = kernel_w_ = unpool_param.kernel_size();
  } else {
    kernel_h_ = unpool_param.kernel_h();
    kernel_w_ = unpool_param.kernel_w();
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!unpool_param.has_pad_h()) {
    pad_h_ = pad_w_ = unpool_param.pad();
  } else {
    pad_h_ = unpool_param.pad_h();
    pad_w_ = unpool_param.pad_w();
  }
  if (!unpool_param.has_stride_h()) {
    stride_h_ = stride_w_ = unpool_param.stride();
  } else {
    stride_h_ = unpool_param.stride_h();
    stride_w_ = unpool_param.stride_w();
  }
  if (pad_h_ != 0 || pad_w_ != 0) {
    CHECK(this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_AVE
        || this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_MAX)
        << "Padding implemented only for average and max pooling.";
    CHECK_LT(pad_h_, kernel_h_);
    CHECK_LT(pad_w_, kernel_w_);
  }   
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
  pooled_count_ = bottom[0]->count();
  pooled_height_ = bottom[0]->height();
  pooled_width_ = bottom[0]->width();
  unpooled_height_ = static_cast<int>((pooled_height_ - 1) * stride_h_ - 2 * pad_h_ + kernel_h_);
  unpooled_width_  = static_cast<int>((pooled_width_  - 1) * stride_w_ - 2 * pad_w_ + kernel_w_);

//   LOG(INFO) << "bottom[0]->num_axes() = " << bottom[0]->num_axes();
//   LOG(INFO) << "channels_ = " << channels_;  
//   LOG(INFO) << "pooled_count_ = " << pooled_count_;
//   LOG(INFO) << "pooled_height_ = " << pooled_height_;
//   LOG(INFO) << "pooled_width_ = " << pooled_width_;
//   LOG(INFO) << "unpooled_height_ = " << unpooled_height_;
//   LOG(INFO) << "unpooled_width_ = " << unpooled_width_;
//   LOG(INFO) << "pad_h_ = " << pad_h_;
//   LOG(INFO) << "pad_w_ = " << pad_w_;
//   LOG(INFO) << "kernel_h_ = " << kernel_h_;
//   LOG(INFO) << "kernel_w_ = " << kernel_w_;
//   LOG(INFO) << "stride_h_ = " << stride_h_;
//   LOG(INFO) << "stride_w_ = " << stride_w_;

  top[0]->Reshape(bottom[0]->num(), channels_, unpooled_height_,
      unpooled_width_);
}

template <typename Dtype>
void UnpoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LOG(INFO) << "fwd begin ";

//   const Dtype* bottom_data = bottom[0]->cpu_data();
//   const Dtype* mask = bottom[1]->cpu_data();
//   Dtype* top_data = top[0]->mutable_cpu_data();
//   caffe_set(top[0]->count(), Dtype(0), top_data);
// 
//   for (int n = 0; n < bottom[0]->num(); ++n) {
//     for (int c = 0; c < channels_; ++c) {
//       for (int idx = 0; idx < pooled_count_; ++idx) {
//         top_data[static_cast<int>(mask[idx])] = bottom_data[idx];
//       }
//         // compute offset
//         bottom_data += bottom[0]->offset(0, 1);
//         top_data += top[0]->offset(0, 1);
//         mask += bottom[0]->offset(0, 1);
//     }
//   }


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
// 	  LOG(INFO) << "top_data[" << unpooled_index <<"] = " << top_data[unpooled_index];
// 	  LOG(INFO) << "bottom_data[" << pooled_index <<"] = " << bottom_data[pooled_index];
        }
      }
      bottom_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
      mask += bottom[0]->offset(0, 1);
    }
  }
  LOG(INFO) << "fwd OK ";
}

template <typename Dtype>
void UnpoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  LOG(INFO) << "backwd begin ";
  
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* bottom_mask = bottom[1]->cpu_data();
  Dtype* bottom_data_diff = bottom[0]->mutable_cpu_diff();
  Dtype* bottom_mask_diff = bottom[1]->mutable_cpu_diff();
  
  const int bottom_count = bottom[0]->count();
  caffe_set(bottom_count, Dtype(0), bottom_data_diff);
  caffe_set(bottom_count, Dtype(0), bottom_mask_diff);
  
  // copy bottom_mask to bottom_mask_diff TODO: probably unnecessary!!
  for (int i = 0; i < bottom_count; ++i) {
    bottom_mask_diff[i] = bottom_mask[i];
  }
  
  // put data diffs to the argmax positions
  for (int i = 0; i < bottom_count; ++i) {
    const int index = bottom_mask[i];
    bottom_data_diff[i] += top_diff[index];
  }
  
//   for (int n = 0; n < top[0]->num(); ++n) {
//     for (int c = 0; c < channels_; ++c) {
//       for (int ph = 0; ph < pooled_height_; ++ph) {
//         for (int pw = 0; pw < pooled_width_; ++pw) {
//           const int pool_index = ph * pooled_width_ + pw;
// 	  const int index = bottom_mask[h * unpooled_width_ + w];
// 	  bottom_data_diff[pool_index] = top_diff[index];
//         }
//       }
//       // compute offset
//       top_diff += top[0]->offset(0, 1);
//       bottom_data_diff += bottom[0]->offset(0, 1);
//       mask_diff += bottom[0]->offset(0, 1);
//       }
//   }
  LOG(INFO) << "backwd OK ";

}


#ifdef CPU_ONLY
STUB_GPU(UnpoolingLayer);
#endif

INSTANTIATE_CLASS(UnpoolingLayer);
REGISTER_LAYER_CLASS(Unpooling);

}  // namespace caffe
