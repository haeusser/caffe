#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {
  
  namespace HuberL1 {

  template <typename Dtype>
  __global__ void ComputeSign(const int n, const Dtype* in, Dtype* out) {
    CUDA_KERNEL_LOOP(index, n) {
      out[index] = in[index] > 0 ? Dtype(1) : Dtype(-1);
    }
  } 

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
  __global__ void KillMasked(const int n, const Dtype* in, Dtype* out) {
    CUDA_KERNEL_LOOP(index, n) {
      out[index] = in[index]>0.5 ? out[index] : Dtype(0);
    }
  }
  
  /**
   * @brief Square data within the range [-0.5, 0.5], and bring everything
   *        outside that range by 0.25 closer to zero (== Turn a linear 
   *        function into something whose absolute is the Huber-L1 norm.
   * 
   * @param KEEP_SIGN (template parameter) IFF TRUE, pixels with negative values 
   *                  will be assigned the _negated_ square of their value
   * @param n Number of data points
   * @param data Input/output data pointer
   */
  template <typename Dtype, bool KEEP_SIGN>
  __global__ void Huberize(const int n, Dtype* data) {
    const Dtype huber_threshold = (Dtype)0.5;
    CUDA_KERNEL_LOOP(index, n) {
      Dtype& value = data[index];
      if (-huber_threshold < value and value < huber_threshold) {
        if (KEEP_SIGN and value < (Dtype)0) {
          value = -value*value;
        } else {
          value = value*value;
        }
      } else {
        if (KEEP_SIGN and value < (Dtype)0) {
          value += (Dtype)0.25;
        } else {
          value -= (Dtype)0.25;
        }
      }
    }
  }
  
  /**
   * @brief Compute gradient of a pseudo-Huberized L1 function (result of
   *        HuberL1::Huberize<...,true><<<...>>>(...)
   * 
   * @param KEEP_SIGN (template parameter) IFF TRUE, pixels with negative values 
   *                  will be assigned their real gradient
   * @param n Number of data points
   * @param data Input/output data pointer
   */
  template <typename Dtype, bool KEEP_SIGN>
  __global__ void Hubergradient(const int n, Dtype* data) {
    const Dtype huber_threshold = (Dtype)0.5;
    CUDA_KERNEL_LOOP(index, n) {
      Dtype& value = data[index];
      if (-huber_threshold < value and value < huber_threshold) {
        if (KEEP_SIGN) {
          value = (Dtype)2*value;
        } else {
          value = (Dtype)2*(value<(Dtype)0 ? -value : value);
        }
      } else {
        if (KEEP_SIGN and value < (Dtype)0) {
          value = (Dtype)-1;
        } else {
          value = (Dtype)1;
        }
      }
    }
  }
  
  }  // namespace HuberL1


template <typename Dtype>
void HuberL1LossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
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
  HuberL1::FindNotNaNs<Dtype><<<CAFFE_GET_BLOCKS(count), 
                                CAFFE_CUDA_NUM_THREADS>>>(
                                  count, 
                                  diffptr->gpu_data(), 
                                  mask_.mutable_gpu_data());
  CUDA_POST_KERNEL_CHECK;
  if (this->layer_param_.huber_l1_loss_param().normalize_by_num_entries()) {
    caffe_gpu_dot(count, mask_.gpu_data(), mask_.gpu_data(), &normalize_coeff_);
    normalize_coeff_ /= mask_.channels();
  } else {
    normalize_coeff_ = num;
  }
  
  // set NaNs to zero
  HuberL1::KillNaNs<Dtype><<<CAFFE_GET_BLOCKS(count), 
                             CAFFE_CUDA_NUM_THREADS>>>(
                                count, 
                                diffptr->mutable_gpu_data(), 
                                diffptr->mutable_gpu_data());
  CUDA_POST_KERNEL_CHECK;
  
  if (this->layer_param_.huber_l1_loss_param().l2_per_location()) {
    /// TODO
    square_layer_->Forward(diff_top_vec_, square_top_vec_);
    sum_layer_->Forward(square_top_vec_, sum_top_vec_);
    sqrt_layer_->Forward(sum_top_vec_, sqrt_top_vec_);
    HuberL1::Huberize<Dtype,false><<<CAFFE_GET_BLOCKS(sqrt_output_.count()),
                                     CAFFE_CUDA_NUM_THREADS>>>(
                                        sqrt_output_.count(),
                                        sqrt_output_.mutable_gpu_data());
    CUDA_POST_KERNEL_CHECK;
    caffe_gpu_dot(sqrt_output_.count(), 
                  sqrt_output_.gpu_data(), 
                  sign_.gpu_data(), 
                  &dot);
  }
  else {
    /// TODO
    HuberL1::ComputeSign<Dtype><<<CAFFE_GET_BLOCKS(count), 
                                  CAFFE_CUDA_NUM_THREADS>>>(
                                    count, 
                                    diffptr->gpu_data(), 
                                    sign_.mutable_gpu_data());
    CUDA_POST_KERNEL_CHECK;
    HuberL1::Huberize<Dtype,true><<<CAFFE_GET_BLOCKS(count), 
                                    CAFFE_CUDA_NUM_THREADS>>>(
                                        count, 
                                        diffptr->mutable_gpu_data());
    CUDA_POST_KERNEL_CHECK;
    caffe_gpu_dot(count, diffptr->gpu_data(), sign_.gpu_data(), &dot); 
  }
  loss = dot / normalize_coeff_; 
  top[0]->mutable_cpu_data()[0] = loss;  
}

template <typename Dtype>
void HuberL1LossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                           const vector<bool>& propagate_down, 
                                           const vector<Blob<Dtype>*>& bottom)
{
  bool prop_down = propagate_down[0];
  if(bottom.size() > 1) prop_down |= propagate_down[1];
  
  Blob<Dtype> *diffptr = diff_top_vec_[0];
  
  if (prop_down) {
    const Dtype alpha = top[0]->cpu_diff()[0] / normalize_coeff_;
    if (this->layer_param_.huber_l1_loss_param().l2_per_location()) {
      /// TODO
      vector<bool> prop_down(1,true);
      HuberL1::Hubergradient<Dtype, true><<<CAFFE_GET_BLOCKS(sqrt_output_.count()),
                                            CAFFE_CUDA_NUM_THREADS>>>(
                                              sqrt_output_.count(),
                                              sqrt_output_.mutable_gpu_data());
      CUDA_POST_KERNEL_CHECK;
      caffe_gpu_axpby(sqrt_output_.count(), 
                      alpha, 
                      sign_.gpu_data(), 
                      Dtype(0), 
                      sqrt_output_.mutable_gpu_diff());
      sqrt_layer_->Backward(    sqrt_top_vec_, prop_down,    sum_top_vec_);
      sum_layer_->Backward(      sum_top_vec_, prop_down, square_top_vec_);
      square_layer_->Backward(square_top_vec_, prop_down,   diff_top_vec_);
    }
    else {
      /// TODO
      /**
       *                                |
       * L1:                            | Huber-L1:
       *   .        ¦        .          |     .        ¦        .
       *     .      ¦      .            |       .      ¦      .
       *       .    ¦    .              |         .    ¦    .
       *         .  ¦  .                |           - _¦_ -
       * -----------+-----------        |   ----------------------
       *            ¦                   |           
       *            ¦                   |           |-----| <-- quadratic
       *                                |
       * Gradient:                      | Gradient:
       *            ¦                   |              ¦ 
       *            ¦___________   _+1  |              ¦  ________  _+1
       *            |                   |              ¦ /
       *            |                   |              ¦/
       * -----------+-----------        |   -----------/---------- 
       *            |                   |             /¦ 
       * ___________|              _-1  |   _________/ ¦            _-1
       *            ¦                   |              ¦ 
       *            ¦                   |              ¦ 
       *                                |
       * 
       * ! Outside the square region, Huber-L1 is (L1 - 0.25) !
       */
      HuberL1::Hubergradient<Dtype, true><<<CAFFE_GET_BLOCKS(diffptr->count()),
                                            CAFFE_CUDA_NUM_THREADS>>>(
                                              diffptr->count(),
                                              diffptr->mutable_gpu_data());
      CUDA_POST_KERNEL_CHECK;
      /// y := alpha*x+beta*y
      caffe_gpu_axpby(diffptr->count(),
                      alpha,                       /*alpha*/
                      diffptr->gpu_data(),         /*x*/ 
                      Dtype(0),                    /*beta*/
                      diffptr->mutable_gpu_diff()  /*y*/
                     );
    }
    
    HuberL1::KillMasked<Dtype><<<CAFFE_GET_BLOCKS(diffptr->count()),
                                 CAFFE_CUDA_NUM_THREADS>>>(
                                    diffptr->count(), 
                                    mask_.gpu_data(), 
                                    diffptr->mutable_gpu_diff());
    CUDA_POST_KERNEL_CHECK;
    
    if(bottom.size() > 1) {
        diff_layer_->Backward(diff_top_vec_, propagate_down, bottom);
    }
  }
  
}

INSTANTIATE_LAYER_GPU_FUNCS(HuberL1LossLayer);

}  // namespace caffe
