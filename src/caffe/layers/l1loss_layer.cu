#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ComputeSign(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ? Dtype(1) : Dtype(-1);
  }
} 

// TODO maybe change the way of detecting NaNs

template <typename Dtype>
__global__ void FindNotNaNs(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index]==in[index] ? Dtype(1) : Dtype(0);
  }
} 

template <typename Dtype>
__global__ void KillNaNs(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index]==in[index] ? in[index] : Dtype(0);
  }
}
  
template <typename Dtype>
void L1LossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
  
  Blob<Dtype> *diffptr = diff_top_vec_[0];
  
  Dtype dot, loss;
  if(bottom.size() > 1) {
    diff_layer_->Forward(bottom, diff_top_vec_);
  }
  
  // if necessary, compute the number of not-NaNs
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  if (this->layer_param_.l1_loss_param().normalize_by_num_entries()) {
    FindNotNaNs<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, diffptr->gpu_data(), sign_.mutable_gpu_data());
    CUDA_POST_KERNEL_CHECK;
    caffe_gpu_dot(count, sign_.gpu_data(), sign_.gpu_data(), &normalize_coeff_);
    normalize_coeff_ /= sign_.channels();
    if (this->layer_param_.l1_loss_param().l2_per_location())  
      caffe_gpu_set(sign_.count()/sign_.channels(), Dtype(1), sign_.mutable_gpu_data());
  } else
    normalize_coeff_ = num;
  
  // set NaNs to zero
  KillNaNs<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, diffptr->gpu_data(), diffptr->mutable_gpu_data());
  CUDA_POST_KERNEL_CHECK;
  
  if (this->layer_param_.l1_loss_param().l2_per_location()) {
    square_layer_->Forward(diff_top_vec_, square_top_vec_);
    sum_layer_->Forward(square_top_vec_, sum_top_vec_);
    sqrt_layer_->Forward(sum_top_vec_, sqrt_top_vec_);
    caffe_gpu_dot(sqrt_output_.count(), sqrt_output_.gpu_data(), sign_.gpu_data(), &dot);
  }
  else {    
    ComputeSign<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, diffptr->gpu_data(), sign_.mutable_gpu_data());
    CUDA_POST_KERNEL_CHECK;
    caffe_gpu_dot(count, diffptr->gpu_data(), sign_.gpu_data(), &dot); 
  }
  loss = dot / normalize_coeff_; 
  top[0]->mutable_cpu_data()[0] = loss;  
}

template <typename Dtype>
void L1LossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
  
  bool prop_down = propagate_down[0];
  if(bottom.size() > 1) prop_down |= propagate_down[1];
  
  Blob<Dtype> *diffptr = diff_top_vec_[0];
  
  if (prop_down) {
    const Dtype alpha = top[0]->cpu_diff()[0] / normalize_coeff_;
    if (this->layer_param_.l1_loss_param().l2_per_location()) {
      vector<bool> prop_down(1,true);
      caffe_gpu_axpby(sqrt_output_.count(), alpha, sign_.gpu_data(),                   
          Dtype(0), sqrt_output_.mutable_gpu_diff());
      sqrt_layer_->Backward(sqrt_top_vec_, prop_down, sum_top_vec_);
      sum_layer_->Backward(sum_top_vec_, prop_down, square_top_vec_);
      square_layer_->Backward(square_top_vec_, prop_down, diff_top_vec_);
    }
    else {    
      caffe_gpu_axpby(diffptr->count(), alpha, sign_.gpu_data(), 
          Dtype(0), diffptr->mutable_gpu_diff());
    }
    
    if(bottom.size() > 1) {
        diff_layer_->Backward(diff_top_vec_, propagate_down, bottom);
    }
  }
  
}

INSTANTIATE_LAYER_GPU_FUNCS(L1LossLayer);

}  // namespace caffe
