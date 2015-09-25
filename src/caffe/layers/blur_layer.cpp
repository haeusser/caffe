// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

#define min(a,b) ((a<b)?a:b)

namespace caffe {

template <typename Dtype>
void BlurLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    if(this->layer_param().blur_param().type() != BlurParameter_BlurType_GAUSSIAN)
        LOG(FATAL) << "BlurLayer: only GAUSSIAN is supported for now";
}

template <typename Dtype>
void BlurLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
 
  this->layer_param_.set_reshape_every_iter(false);
  LOG(WARNING) << "BlurLayer only runs Reshape on setup";
  
  CHECK_EQ(bottom.size(), 1);
  CHECK_EQ(top.size(), 1);
  
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int width = bottom[0]->width();
  int height = bottom[0]->height();

  top[0]->Reshape(num, channels, height, width);
}

template <typename Dtype>
void BlurLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  LOG(FATAL) << "BlurLayer: CPU Forward not yet implemented.";
}

template <typename Dtype>
void BlurLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  LOG(FATAL) << "BlurLayer cannot do backward.";
}

INSTANTIATE_CLASS(BlurLayer);
REGISTER_LAYER_CLASS(Blur);

}  // namespace caffe
