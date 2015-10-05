#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void BlobFromScalarLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    CHECK_EQ(bottom.size(), 1) << "Need one input";
    CHECK_EQ(top.size(), 1) << "Need one output";
}

template <typename Dtype>
void BlobFromScalarLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    CHECK_EQ(bottom[0]->num_axes(), 0) << "Input must be a scalar (dim=0)";
    top[0]->Reshape(1,1,1,1);
}

template <typename Dtype>
void BlobFromScalarLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
    *(top[0]->mutable_cpu_data()) = *(bottom[0]->mutable_cpu_data());
}

template <typename Dtype>
void BlobFromScalarLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
    *(bottom[0]->mutable_cpu_diff()) = *(top[0]->mutable_cpu_diff());
}

#ifdef CPU_ONLY
STUB_GPU(BlobFromScalarLayer);
#endif

INSTANTIATE_CLASS(BlobFromScalarLayer);
REGISTER_LAYER_CLASS(BlobFromScalar);

}  // namespace caffe
