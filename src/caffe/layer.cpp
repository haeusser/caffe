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
void Layer<Dtype>::UpdateActiveness(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
  /// At least one top blob active?
  bool some_top_blob_active = false;
  for (int top_id = 0; top_id < top.size(); ++top_id) 
    some_top_blob_active |= top[top_id]->GetActivenessFlag();
  
  /// Monitor changes in the activeness flags of all top blobs and change
  /// own activeness if necessary
  switch (activeness_) {
    case ACTIVE: {
      /// Stay active if any top blog is active...
      /// ...else become inactive
      if (not some_top_blob_active) {
        activeness_ = BECOMING_INACTIVE;
        LOG(INFO) << "Preparing layer " << layer_param_.name() << " for inactivity";
      }
      break;
    }
    case BECOMING_INACTIVE: {
      /// Cancel this layer's deactivation if any top blob is active...
      if (some_top_blob_active) {
        activeness_ = ACTIVE;
        LOG(INFO) << "Cancelling deactivation of layer " << layer_param_.name();
      }
      /// ...else switch to full inactivity
      else {
        activeness_ = INACTIVE;
        LOG(INFO) << "Layer " << layer_param_.name() << " is now inactive";
      }
      break;
    }
    case INACTIVE: {
      /// Reactivate if any top blob is active, else stay inactive
      if (some_top_blob_active) {
        activeness_ = ACTIVE;
        LOG(INFO) << "Reactivating layer " << layer_param_.name();
      }
      break;
    }
    default: {
      LOG(ERROR) << "Invalid value (" << activeness_ << ") for activeness_";
    }
  }

      
  /// The layer's actions depend on its NEW activeness state
  switch (activeness_) {
    case ACTIVE: {
      /// Set activeness for bottom blobs
      for (int bot_id = 0; bot_id < bottom.size(); ++bot_id) 
        bottom[bot_id]->SetActivenessFlag(true);
      break;
    }
    case BECOMING_INACTIVE: {
      /// Set gradients to zero iff BECOMING_INACTIVE
      for (int blob_id = 0; blob_id < blobs_.size(); ++blob_id) blobs_[blob_id]->scale_diff(0);
      for (int blob_id = 0; blob_id < bottom.size(); ++blob_id) bottom[blob_id]->scale_diff(0);
      
      /// Propagate flag from top to bottom layers      
      for (int bot_id = 0; bot_id < bottom.size(); ++bot_id) 
        bottom[bot_id]->SetActivenessFlag(false);
      break;
    }
    case INACTIVE: {
      /// Flag bottom blobs for inactiveness
      for (int bot_id = 0; bot_id < bottom.size(); ++bot_id) 
        bottom[bot_id]->SetActivenessFlag(false);
      
      break;
    }
    default: {
      LOG(ERROR) << "Invalid value (" << activeness_ << ") for activeness_";
    }
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
          
          bool activeness = (new_loss_weight > 0);
          for (int top_id = 0; top_id < top.size(); ++top_id) 
            top[top_id]->SetActivenessFlag(activeness);
          if (not activeness) {
            LOG(INFO) << "Layer " << this->layer_param_.name()
                      << " has marked itself for inactiveness.";
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
