#ifndef CAFFE_UTIL_DB_BINARYDB_HPP
#define CAFFE_UTIL_DB_BINARYDB_HPP

/// System/STL
#include <fstream>
#include <map>
#include <queue>
#include <string>
#include <vector>
/// Boost
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
/// Caffe/local files
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/blob.hpp"

namespace caffe { 
namespace db {

 
  
template <typename Dtype>
class BinaryDB {
public:
  BinaryDB();
  ~BinaryDB();
  
  void Open(
        const string& source, 
        const LayerParameter& param);
  
  void Close();
  
  int get_num_samples() { return num_samples_; };
  
  void get_sample(
        int index, 
        vector<Blob<Dtype>*>* dst, 
        int* compressed_size);
  
//   void read_binstream(
//         std::ifstream* binstream, 
//         BinaryDB_DataEncoding data_encoding, 
//         long int N, 
//         Dtype* out, 
//         int* n_read);

private:
  
  struct Entry {
    int binfile_idx;
    long int byte_offset;
    BinaryDB_DataEncoding data_encoding;
  };
  
  struct ReadTask {
    ReadTask( Entry& entry_ref,
              std::ifstream* binstream_ptr,
              long int N,
              Dtype* dst_ptr,
              unsigned char* entry_buffer)
    : entry_ref(entry_ref),
      binstream_ptr(binstream_ptr),
      N(N),
      dst_ptr(dst_ptr),
      n_read(0),
      entry_buffer(entry_buffer)
    {}
    
    Entry& entry_ref;
    std::ifstream* binstream_ptr;
    long int N;
    Dtype* dst_ptr;
    int n_read;
    unsigned char* entry_buffer;
    
    /// DEBUG
    void debug(int s, int i)
    { sample=s; index=i; }
    int sample;
    int index;
    /// DEBUG
  };
  std::queue<ReadTask*> undone_tasks;
  std::queue<ReadTask*> done_tasks;
  boost::mutex undone_tasks__LOCK;
  boost::mutex done_tasks__LOCK;
  std::vector<boost::thread*> worker_threads;
  std::vector<unsigned char*> entry_buffers_;
  bool running;
  
  void check_flag(
        unsigned char* buffer);
  void process_readtask(
        ReadTask* task_ptr);
  void worker_thread_loop();
  
  
  typedef vector<Entry> Sample;
  
  int top_num_;
  int sample_variants_num_;
  int num_samples_;
  int num_binfiles_;
  
  int entry_buffer_size_;

  std::vector<Sample> samples_;
  std::vector<std::vector<int> > entry_dimensions_;
  std::vector<std::string> binfiles_;
//   std::vector<boost::shared_ptr<std::ifstream> > binstreams_;
  std::map<int, std::map<int, std::ifstream*> > binstreams_;
  std::vector<int> permutation_;  
  
  
  /**
   * string.split() (from http://stackoverflow.com/a/236803)
   */  
  std::vector<std::string>& debug_m_split(const std::string &s,
                                  char delim,
                                  std::vector<std::string>& elems)
  {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
      elems.push_back(item);
    }
    return elems;
  }

  std::vector<std::string> debug_m_split(const std::string& s, 
                                char delim) 
  {
    std::vector<std::string> elems;
    debug_m_split(s, delim, elems);
    return elems;
  }
  
};


}  // namespace db
}  // namespace caffe

#endif  // CAFFE_UTIL_DB_BINARYDB_HPP
