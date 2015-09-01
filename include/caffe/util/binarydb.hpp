#ifndef CAFFE_UTIL_DB_BINARYDB_HPP
#define CAFFE_UTIL_DB_BINARYDB_HPP

#include <string>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe { namespace db {

template <typename Dtype>
class BinaryDB {
 public:
  BinaryDB() {
    
  }
  ~BinaryDB() { Close(); }
  void Open(const string& source, const LayerParameter& param);
  
  void Close();
  
  int get_num_samples();
  
  void get_sample(int index, vector<Blob<Dtype>*>* dst);

 private:
};

}  // namespace db
}  // namespace caffe

#endif  // CAFFE_UTIL_DB_BINARYDB_HPP
