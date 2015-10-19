#include <vector>

#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {
  
  namespace LpqLayer__kernels {

    /**
    * 
    */
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
    __global__ void KillMasked(const int n, const Dtype* in, Dtype* out) {
      CUDA_KERNEL_LOOP(index, n) {
        out[index] = in[index] > Dtype(0.5) ? out[index] : Dtype(0);
      }
    }

    template <typename Dtype>
    __global__ void KillMaskedAcrossChannels(const int n, const int width_height, const Dtype* in, Dtype* out) {
      CUDA_KERNEL_LOOP(index, n) {
        const int mask_idx = index % width_height;
        out[index] = in[mask_idx] > Dtype(0.5) ? out[index] : Dtype(0);
      }
    }

    template <typename Dtype>
    __global__ void MaskPlateauValues(const int n, const Dtype* in, Dtype* out, Dtype plateau) {
      CUDA_KERNEL_LOOP(index, n) {
        if(fabs(in[index]) < plateau) out[index] = Dtype(0); // Mask out plateau values and keep other as is
      }
    } 

    template <typename Dtype>
    __global__ void MaskPlateauValuesInitial(const int n, const Dtype* in, Dtype* out, Dtype plateau) {
      CUDA_KERNEL_LOOP(index, n) {
        out[index] = (fabs(in[index]) < plateau) ? Dtype(0) : Dtype(1);
      }
    } 

  }  // namespace LpqLayer__kernels


  template <typename Dtype>
  void LpqLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top)
  {
    /// Check and reset p/q parameters
    {
      /// Get current iteration
      Net<Dtype> *net = this->GetNet();
      unsigned int current_iteration = net->iter();
      
      /// Discard all old schedule steps
      while (schedule_.size() > 0 and 
             current_iteration > schedule_.front().start_iter)
      {
        schedule_.pop();
      }
      /// If there is a schedule step left, check if we have reached it
      if (schedule_.size() > 0 and 
          current_iteration == schedule_.front().start_iter) 
      {
        LOG(INFO) << "Lpq loss layer: Iteration " << current_iteration
                  << ", switching to p=" << schedule_.front().p
                  << ", q=" << schedule_.front().q;
        
        /// Reset p-power layer
        p_top_vec_.clear();
        p_top_vec_.push_back(&p_output_);
        LayerParameter p_param;
        p_param.mutable_power_param()->set_power(schedule_.front().p);
        p_layer_.reset(new PowerLayer<Dtype>(p_param));
        p_layer_->SetUp(diff_top_vec_, p_top_vec_);
        /// Reset q-power layer
        q_top_vec_.clear();
        q_top_vec_.push_back(&q_output_);
        LayerParameter q_param;
        q_param.mutable_power_param()->set_power(schedule_.front().q);
        q_param.mutable_power_param()->set_shift(
            this->layer_param_.l1_loss_param().epsilon());
        q_layer_.reset(new PowerLayer<Dtype>(q_param));
        q_layer_->SetUp(sum_top_vec_, q_top_vec_);
        /// Discard schedule step
        schedule_.pop();
      }
    }
    
    
    Blob<Dtype> *diffptr = diff_top_vec_[0];
    
    Dtype dot, loss;
    if(bottom.size() > 1) {
      diff_layer_->Forward(bottom, diff_top_vec_);
    }
    
    // if necessary, compute the number of not-NaNs
    int count = bottom[0]->count();
    int num = bottom[0]->num();
    LpqLayer__kernels::FindNotNaNs<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, diffptr->gpu_data(), mask_.mutable_gpu_data());
    cudaDeviceSynchronize();
    CUDA_POST_KERNEL_CHECK;
    
    if (this->layer_param_.l1_loss_param().normalize_by_num_entries()) {    
      caffe_gpu_dot(count, mask_.gpu_data(), mask_.gpu_data(), &normalize_coeff_);
      normalize_coeff_ /= mask_.channels();
    } else {
      normalize_coeff_ = num;
    }
    
    if (this->layer_param_.l1_loss_param().l2_per_location()) {
      // set masked (NaNs only) to zero
      LpqLayer__kernels::KillMasked<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, mask_.gpu_data(), diffptr->mutable_gpu_data());
      cudaDeviceSynchronize();
      CUDA_POST_KERNEL_CHECK;
      
      p_layer_->Forward(diff_top_vec_, p_top_vec_);
      sum_layer_->Forward(p_top_vec_, sum_top_vec_);
      
      // Mask plateau in summed blob (only one channel):
      if(this->layer_param_.l1_loss_param().plateau() > 0) {
        float plateau_val_squared = this->layer_param_.l1_loss_param().plateau() * this->layer_param_.l1_loss_param().plateau();
        LpqLayer__kernels::MaskPlateauValuesInitial<Dtype><<<CAFFE_GET_BLOCKS(sum_output_.count()), CAFFE_CUDA_NUM_THREADS>>>(
            sum_output_.count(), sum_output_.gpu_data(), plateau_l2_.mutable_gpu_data(), plateau_val_squared);
        cudaDeviceSynchronize();
        CUDA_POST_KERNEL_CHECK;
        
        LpqLayer__kernels::KillMasked<Dtype><<<CAFFE_GET_BLOCKS(sum_output_.count()), CAFFE_CUDA_NUM_THREADS>>>(
              sum_output_.count(), plateau_l2_.gpu_data(), sum_output_.mutable_gpu_data());
        cudaDeviceSynchronize();
        CUDA_POST_KERNEL_CHECK;
      }
      
      q_layer_->Forward(sum_top_vec_, q_top_vec_);
      // Note sign_ is set to all ones in Reshape
      caffe_gpu_dot(q_output_.count(), q_output_.gpu_data(), sign_.gpu_data(), &dot);
    }
    else {    
      // Mask plateau:
      if(this->layer_param_.l1_loss_param().plateau() > 0) {
        LpqLayer__kernels::MaskPlateauValues<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, diffptr->gpu_data(), mask_.mutable_gpu_data(), this->layer_param_.l1_loss_param().plateau());
        CUDA_POST_KERNEL_CHECK;
      }
      
      //mask_.print("MASK2");
      
      // set masked (NaNs, plateau) to zero
      LpqLayer__kernels::KillMasked<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, mask_.gpu_data(), diffptr->mutable_gpu_data());
      CUDA_POST_KERNEL_CHECK;
      
      LpqLayer__kernels::ComputeSign<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, diffptr->gpu_data(), sign_.mutable_gpu_data());
      CUDA_POST_KERNEL_CHECK;
      caffe_gpu_dot(count, diffptr->gpu_data(), sign_.gpu_data(), &dot); 
    }
    loss = dot / normalize_coeff_; 
    top[0]->mutable_cpu_data()[0] = loss;
  }

  template <typename Dtype>
  void LpqLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
  {  
    bool prop_down = propagate_down[0];
    if(bottom.size() > 1) prop_down |= propagate_down[1];
    
    Blob<Dtype> *diffptr = diff_top_vec_[0];
    
    if (prop_down) {
      const Dtype alpha = top[0]->cpu_diff()[0] / normalize_coeff_;
      if (this->layer_param_.l1_loss_param().l2_per_location()) {
        vector<bool> prop_down(1,true);
        caffe_gpu_axpby(q_output_.count(), alpha, sign_.gpu_data(),                   
            Dtype(0), q_output_.mutable_gpu_diff());
        q_layer_->Backward(q_top_vec_, prop_down, sum_top_vec_);
        
        if(this->layer_param_.l1_loss_param().plateau() > 0) {
          LpqLayer__kernels::KillMasked<Dtype><<<CAFFE_GET_BLOCKS(sum_output_.count()), CAFFE_CUDA_NUM_THREADS>>>(
                sum_output_.count(), plateau_l2_.gpu_data(), sum_output_.mutable_gpu_diff());
          cudaDeviceSynchronize();
          CUDA_POST_KERNEL_CHECK;
        }
        
        sum_layer_->Backward(sum_top_vec_, prop_down, p_top_vec_);
        p_layer_->Backward(p_top_vec_, prop_down, diff_top_vec_);
        
      
      }
      else {    
        caffe_gpu_axpby(diffptr->count(), alpha, sign_.gpu_data(), 
            Dtype(0), diffptr->mutable_gpu_diff());
      }
      
      LpqLayer__kernels::KillMasked<Dtype><<<CAFFE_GET_BLOCKS(diffptr->count()), CAFFE_CUDA_NUM_THREADS>>>(
          diffptr->count(), mask_.gpu_data(), diffptr->mutable_gpu_diff());
      CUDA_POST_KERNEL_CHECK;
      
      if(bottom.size() > 1) {
          diff_layer_->Backward(diff_top_vec_, propagate_down, bottom);
      }
    }
    
  }

  INSTANTIATE_LAYER_GPU_FUNCS(LpqLossLayer);

}  // namespace caffe
