#ifndef CAFFE_UTIL_DB_BINARYDB_HPP
#define CAFFE_UTIL_DB_BINARYDB_HPP

#include <string>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/blob.hpp"

namespace caffe { namespace db {

 
  
template <typename Dtype>
class BinaryDB {
 public:
  BinaryDB() {
    num_samples_ = 0;
  }
  ~BinaryDB() { Close(); }
  void Open(const string& source, const LayerParameter& param);
  
  void Close();
  
  int get_num_samples() { return num_samples_; };
  
  void get_sample(int index, vector<Blob<Dtype>*>* dst);
  
  void read_binstream(std::ifstream* binstream, BinaryDB_DataEncoding data_encoding, long int N, Dtype* out);

 private:
  struct Entry {
    int binfile_idx;
    long int byte_offset;
    BinaryDB_DataEncoding data_encoding;
  };  
  
  typedef vector<Entry> Sample;
  
  int top_num_;
  int sample_variants_num_;
  int num_samples_;
  int num_binfiles_;
  
  vector<Sample> samples_;
  vector<vector<int> > entry_dimensions_;
  vector<std::string> binfiles_;
  vector<boost::shared_ptr<std::ifstream> > binstreams_;
  vector<int> permutation_;  
  
};

}  // namespace db
}  // namespace caffe

#endif  // CAFFE_UTIL_DB_BINARYDB_HPP
