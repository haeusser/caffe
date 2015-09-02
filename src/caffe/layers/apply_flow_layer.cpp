// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <vector>
#include <cmath>

#include "google/protobuf/descriptor.h"
#include "google/protobuf/descriptor.h"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/math_functions.hpp"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

#include <iostream>
#include <fstream>
#include <omp.h>

using std::max;

namespace caffe {

template <typename Dtype>
void ApplyFlowLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void ApplyFlowLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 2) << "Apply flow layer takes two input blobs: image and flow.";
  CHECK_EQ(top.size(), 1) << "Apply flow layer outputs one blob.";
  
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  
  CHECK_EQ(num, bottom[1]->num()) << "Num of the inputs should be the same";
  CHECK_EQ(2, bottom[1]->channels()) << "Flow should have 2 channels: x-flow and y-flow";
  CHECK_EQ(width, bottom[1]->width()) << "Width of the inputs should be the same";
  CHECK_EQ(height, bottom[1]->height()) << "Height of the inputs should be the same";  
  
   // = Allocate output
  top[0]->Reshape(num, channels, height, width);
    
}

template <typename Dtype>
void ApplyFlowLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
  LOG(FATAL) << "Forward CPU Augmentation not implemented.";

}

#ifdef CPU_ONLY
STUB_GPU(ApplyFlowLayer);
#endif

INSTANTIATE_CLASS(ApplyFlowLayer);
REGISTER_LAYER_CLASS(ApplyFlow);


}  // namespace caffe
