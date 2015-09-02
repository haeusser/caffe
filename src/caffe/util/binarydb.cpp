#include "caffe/util/binarydb.hpp"

#include <string>

namespace caffe { namespace db {

template <typename Dtype>
void BinaryDB<Dtype>::Open(const string& source, const LayerParameter& param) {
  // TODO
  LOG(INFO) << "Opened BinaryDB " << source;
}


template <typename Dtype>
void BinaryDB<Dtype>::Close() {
  //TODO
}

template <typename Dtype>
int BinaryDB<Dtype>::get_num_samples() {
  return 0; // TODO
}

template <typename Dtype>
void BinaryDB<Dtype>::get_sample(int index, vector<Blob<Dtype>*>* dst) {
  //TODO
}


INSTANTIATE_CLASS(BinaryDB);

}  // namespace db
}  // namespace caffe
