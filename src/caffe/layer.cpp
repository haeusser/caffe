#include <boost/thread.hpp>
#include "caffe/layer.hpp"

#include "caffe/net.hpp"

namespace caffe {

template <typename Dtype>
void Layer<Dtype>::InitMutex() {
  forward_mutex_.reset(new boost::mutex());
}

template <typename Dtype>
void Layer<Dtype>::Lock() {
  if (IsShared()) {
    forward_mutex_->lock();
  }
}

template <typename Dtype>
void Layer<Dtype>::Unlock() {
  if (IsShared()) {
    forward_mutex_->unlock();
  }
}


template <typename Dtype>
void Layer<Dtype>::ApplyLossWeightSchedule(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
  // Loss Weight Schedule
  if(next_weightloss_change_at_iter_ >= 0) {
    CHECK(this->net_ != NULL) << "Need a reference to net from Layer if using Loss Weight Schedule";
    int iter = this->net_->iter();
    if(iter >= next_weightloss_change_at_iter_
     && this->layer_param_.has_loss_param()) {
    
      LossParameter lossparam = this->layer_param_.loss_param();
      int schedule_size = lossparam.loss_schedule_iter_size();
      if(schedule_size > 0) {
        int sched_idx = 0;
        float new_loss_weight = -1;
        for(; sched_idx < schedule_size; sched_idx++) {
          if(lossparam.loss_schedule_iter(sched_idx) >= iter) {
            new_loss_weight = lossparam.loss_schedule_lossweight(sched_idx);
            break;
          }
        }
        if(new_loss_weight >= 0) {
          // Change loss weight of blob 0 (currently only one loss output supported)
          LOG(INFO) << "Changing loss weight of layer " << this->layer_param_.name() << " to " << new_loss_weight << " (sched_idx = " << sched_idx << ")";
          layer_param_.set_loss_weight(0, new_loss_weight);
          SetLossWeights(top);
          
          if(sched_idx+1 < schedule_size) { // Is there a next entry?
            // Wake up at this iter again to check
            next_weightloss_change_at_iter_ = lossparam.loss_schedule_iter(sched_idx+1);
          }
        }
        
      } else {
        next_weightloss_change_at_iter_ = -1; //Disable schedule checking
      }
    }
  }
}

INSTANTIATE_CLASS(Layer);

}  // namespace caffe
