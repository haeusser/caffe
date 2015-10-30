#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void KittiErrorLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);  
  /*tau = [3 0.05];
  E = abs(D_gt-D_est);
  n_err   = length(find(D_gt>0 & E>tau(1) & E./abs(D_gt)>tau(2)));
  n_total = length(find(D_gt>0));
  d_err = n_err/n_total;
  */
  
}

template <typename Dtype>
void KittiErrorLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);
}

template <typename Dtype>
void KittiErrorLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  
  // pixel error threshold below which prediction is accepted
  Dtype tau1 = this->layer_param_.kitti_error_param().pixel_threshold();
  
  // percentage error below which prediction is accepted
  Dtype tau2 = this->layer_param_.kitti_error_param().percentage_threshold();
    
  //const Dtype tau1 = 3; 
  //const Dtype tau2 = 0.05; // percentage error below which prediction is accepted
  
  const Dtype* pred_data = bottom[0]->cpu_data();
  const Dtype* gt_data = bottom[1]->cpu_data();
  
  int totalpix = 0;
  int errorpix = 0;
  for (int i = 0; i < count; ++i) {
    Dtype gtval = gt_data[i];
    
    if(gtval == gtval) { // is not NaN
      Dtype E = std::abs(pred_data[i] - gtval);
      if(E > tau1 and (E / std::abs(gtval) > tau2)) {
        errorpix++;
      }
      totalpix++;
    }
  }
 
  Dtype loss = Dtype(errorpix) / Dtype(totalpix);
  top[0]->mutable_cpu_data()[0] = loss * 100;
}


#ifdef CPU_ONLY
STUB_GPU(KittiErrorLayer);
#endif

INSTANTIATE_CLASS(KittiErrorLayer);
REGISTER_LAYER_CLASS(KittiError);

}  // namespace caffe
