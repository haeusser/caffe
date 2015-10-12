#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ScaleGradientLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ShareData(*bottom[0]);
}

template <typename Dtype>
void ScaleGradientLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  iter_++;

  Dtype coeff;
  if (discount_coeff_schedule_.initial_coeff() == 0) {
    if (discount_coeff_schedule_.final_coeff() == 0)
      coeff = 0;
    else
      discount_coeff_schedule_.set_initial_coeff(1e-10);
  } else  
    coeff = discount_coeff_schedule_.initial_coeff() *
                exp(( log(discount_coeff_schedule_.final_coeff() /discount_coeff_schedule_.initial_coeff()) ) *
                (Dtype(2) / (Dtype(1) + exp(- 1.0986 * iter_ / discount_coeff_schedule_.half_life())) - Dtype(1)));
                
  caffe_gpu_axpby<Dtype>(top[0]->count(), coeff, top[0]->gpu_diff(),
    Dtype(0), bottom[0]->mutable_gpu_diff());
}

INSTANTIATE_LAYER_GPU_FUNCS(ScaleGradientLayer);

}  // namespace caffe
