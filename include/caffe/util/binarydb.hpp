#ifndef CAFFE_UTIL_DB_BINARYDB_HPP
#define CAFFE_UTIL_DB_BINARYDB_HPP

/// System/STL
#include <fstream>
#include <map>
#include <queue>
#include <sstream>
#include <string>
#include <vector>
/// Boost
#include <boost/chrono.hpp>
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
/// Caffe/local files
#include "caffe/util/blocking_queue.hpp"
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
        int* compressed_size,
        bool wait_for_finish=true);
  
//   void read_binstream(
//         std::ifstream* binstream, 
//         BinaryDB_DataEncoding data_encoding, 
//         long int N, 
//         Dtype* out, 
//         int* n_read);

private:
  
  static uint get_next_task_ID()
  {
    static uint running_ID(0);
    static boost::mutex LOCK;
    
    LOCK.lock();
      ++running_ID;
    LOCK.unlock();
    
    return running_ID;
  }
  
  
  
  struct Ifstream_yielder
  {
    Ifstream_yielder(int max_streams)
    : m_reserve(),
      m_max_streams(max_streams)
    {
      for (unsigned int i = 0; i < m_max_streams; ++i) {
        std::ifstream* tmp = new std::ifstream();
        m_reserve.push(tmp);
      }
    }
    
    ~Ifstream_yielder() 
    {
      while (m_in_use.size() > 0) {
        LOG_EVERY_N(INFO, 1000) << "Waiting for " << m_in_use.size()
                                << " ifstreams to be returned";
        boost::this_thread::sleep_for(boost::chrono::milliseconds(10));
      }
        
      for (unsigned int i = 0; i < m_reserve.size(); ++i) {
        std::ifstream* next = m_reserve.pop();
        if (next) {
          if (next->is_open()) {
            next->close();
          }
          delete next;
        }
      }
    }
    
    std::ifstream* get()
    {
      std::ifstream* next_free = m_reserve.pop();
      {
        boost::lock_guard<boost::mutex> LOCK(m_in_use__LOCK);
        m_in_use.push_back(next_free);
      }
      return next_free;
    }
    
    void put_back(std::ifstream* stream_ptr)
    {
      {
        boost::lock_guard<boost::mutex> LOCK(m_in_use__LOCK);
        /// TEST
        if (std::find(m_in_use.begin(), m_in_use.end(), stream_ptr) == m_in_use.end())
          LOG(FATAL) << "I don't know that pointer";
        /// TEST
        /// http://stackoverflow.com/a/3385251
        m_in_use.erase(std::remove(m_in_use.begin(), m_in_use.end(), stream_ptr),
                       m_in_use.end());
      }
      m_reserve.push_front(stream_ptr);
    }
    
    BlockingQueue<std::ifstream*> m_reserve;
    std::vector<std::ifstream*> m_in_use;
    boost::mutex m_in_use__LOCK;
    int m_max_streams;
  };
  
  struct Ifstream_wrapper
  {
    Ifstream_wrapper(Ifstream_yielder* yielder_ptr)
    : m_yielder_ptr(yielder_ptr)
    {
      if (not m_yielder_ptr)
        LOG(FATAL) << "Ifstream_yielder* is invalid";
      m_stream_ptr = m_yielder_ptr->get();
      if (not m_stream_ptr)
        LOG(FATAL) << "std::ifstream* is invalid";
    }
    
    ~Ifstream_wrapper() 
    {
      if (not m_yielder_ptr)
        LOG(FATAL) << "Ifstream_yielder* is invalid";
      m_yielder_ptr->put_back(m_stream_ptr);
    }
    
    std::ifstream* operator()()
    {
      if (not m_stream_ptr)
        LOG(FATAL) << "std::ifstream* is invalid";
      return m_stream_ptr;
    }
    
    std::ifstream* m_stream_ptr;
    Ifstream_yielder* m_yielder_ptr;
  };
  
//   std::vector<boost::shared_ptr<std::ifstream> > binstreams_;
  std::vector<Ifstream_yielder*> binstream_yielders_;
  //std::vector<int> permutation_;  
  
  struct Entry {
    int binfile_idx;
    long int byte_offset;
    BinaryDB_DataEncoding data_encoding;
  };
  
  struct ReadTask {
    ReadTask( Entry& entry_ref,
              Ifstream_yielder* binstream_yielder_ptr,
              long int N,
              Dtype* dst_ptr)
    : entry_ref(entry_ref),
      binstream_yielder_ptr(binstream_yielder_ptr),
      N(N),
      dst_ptr(dst_ptr),
      n_read(0),
      ID(get_next_task_ID())
    {}
    
    Entry& entry_ref;
    Ifstream_yielder* binstream_yielder_ptr;
    long int N;
    Dtype* dst_ptr;
    int n_read;
    
    /// DEBUG
    void debug(int s, int i)
    { sample=s; index=i; }
    int sample;
    int index;
    uint ID;
    /// DEBUG
  };
  std::queue<ReadTask*> undone_tasks;
  std::vector<uint> in_progress_task_ids;
  std::queue<ReadTask*> done_tasks;
  boost::mutex queues__LOCK;
  std::vector<boost::thread*> worker_threads;
  bool running;
  
  void check_flag(
        unsigned char* buffer);
  void process_readtask(
        ReadTask* task_ptr,
        unsigned char* entry_buffer_);
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

  std::string debug_m_split(const std::string& s,
                            char delim)
  {
    std::vector<std::string> elems;
    debug_m_split(s, delim, elems);
    std::ostringstream oss;
    oss << elems[elems.size()-2] << '/' << elems.back();
    return oss.str();
  }
  
};


}  // namespace db
}  // namespace caffe

#endif  // CAFFE_UTIL_DB_BINARYDB_HPP
