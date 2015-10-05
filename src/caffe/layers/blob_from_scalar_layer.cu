#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {


template <typename Dtype>
void BlobFromScalarLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
    caffe_gpu_memcpy(sizeof(Dtype), bottom[0]->gpu_data(), top[0]->mutable_gpu_data());
}

template <typename Dtype>
void BlobFromScalarLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    caffe_gpu_memcpy(sizeof(Dtype), top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff());
}

INSTANTIATE_LAYER_GPU_FUNCS(BlobFromScalarLayer);

}  // namespace caffe
