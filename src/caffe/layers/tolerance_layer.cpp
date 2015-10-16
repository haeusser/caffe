#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ToleranceLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  const ToleranceParameter& tolerance_param = this->layer_param_.tolerance_param();
  
  CHECK_EQ(bottom[0]->width(), bottom[1]->width()) << "Both bottom blobs must have same width";
  CHECK_EQ(bottom[0]->height(), bottom[1]->height()) << "Both bottom blobs must have same height";
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels()) << "Both bottom blobs must have same number of channels";

  tolerance_radius_ = tolerance_param.tolerance_radius();
  
  CHECK(tolerance_radius_ > 0) << "Must choose positive tolerance radius";
}

template <typename Dtype>
void ToleranceLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  top[0]->ReshapeLike(*(bottom[0]));
}

template <typename Dtype>
void ToleranceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void ToleranceLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;  
}

#ifdef CPU_ONLY
STUB_GPU(ToleranceLayer);
#endif

INSTANTIATE_CLASS(ToleranceLayer);
REGISTER_LAYER_CLASS(Tolerance);

}  // namespace caffe
