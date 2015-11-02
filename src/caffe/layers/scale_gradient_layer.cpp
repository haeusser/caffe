#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {
  
template <typename Dtype>
void ScaleGradientLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  discount_coeff_schedule_ = this->layer_param_.coeff_schedule_param();
  iter_ = 0;
}

template <typename Dtype>
void ScaleGradientLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void ScaleGradientLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ShareData(*bottom[0]);
}

template <typename Dtype>
void ScaleGradientLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int iter = this->GetNet()->GetSolver()->iter();

  Dtype coeff = 1;
  if (discount_coeff_schedule_.initial_coeff() == 0) {
    if (discount_coeff_schedule_.final_coeff() == 0)
      coeff = 0;
    else
      discount_coeff_schedule_.set_initial_coeff(1e-10);
  } else {
    /**
     * c  = coeff
     * c0 = coefficient's scheduled initial value
     * c1 = coefficient's scheduled final value
     * ch = coefficient's scheduled half-life
     * i  = current iteration index
     * 
     *                                       -1.0986 * i / ch
     *             log(c1 / c0) * (2 / (1 + e                ) - 1)
     * c = c0  * e
     */
    coeff = discount_coeff_schedule_.initial_coeff() *
                exp(( log(discount_coeff_schedule_.final_coeff() /discount_coeff_schedule_.initial_coeff()) ) *
                (Dtype(2) / (Dtype(1) + exp(- 1.0986 * iter / discount_coeff_schedule_.half_life())) - Dtype(1)));
  }
  
  if (this->layer_param_.scale_gradient_param().has_normalize_gradient()) {
    Dtype dot;
    if (this->layer_param_.scale_gradient_param().normalize_gradient() == "L2") {
      dot = caffe_cpu_dot(top[0]->count(), top[0]->cpu_diff(), top[0]->cpu_diff());
      coeff /= sqrt(dot + this->layer_param_.scale_gradient_param().epsilon());
    }
    if ((this->GetNet()->GetSolver()->param().display() && (iter % this->GetNet()->GetSolver()->param().display() == 0) && this->layer_param_.scale_gradient_param().verbose()) || this->layer_param_.scale_gradient_param().force_verbose())
      LOG(INFO) << "iter=" << iter << ", norm=" << dot << ", coeff=" << coeff;
  }
                
  caffe_cpu_axpby<Dtype>(top[0]->count(), coeff, top[0]->cpu_diff(),
    Dtype(0), bottom[0]->mutable_cpu_diff());
}
  
#ifdef CPU_ONLY
STUB_GPU(ScaleGradientLayer);
#endif

INSTANTIATE_CLASS(ScaleGradientLayer);
REGISTER_LAYER_CLASS(ScaleGradient);

}  // namespace caffe
