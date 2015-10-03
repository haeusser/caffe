#ifndef CAFFE_UTIL_DB_BINARYDB_HPP
#define CAFFE_UTIL_DB_BINARYDB_HPP

/// System/STL
#include <climits>
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
  
  /**
   * @brief Read a configuration file, parse which data can be read and 
   * instantiate internal structures
   * 
   * @param source Configuration file (the file has already been processed
   *               by the Python backend; this parameter is here only used
   *               for debug prints)
   * @param param Caffe parameters
   */
  void Open(
        const string& source, 
        const LayerParameter& param);
  
  /**
   * @brief Stop worker threads
   */
  void Close();
  
  int get_num_samples() { return num_samples_; };
  
  void get_sample(
        int index, 
        vector<Blob<Dtype>*>* dst, 
        int* compressed_size,
        bool wait_for_finish=true);

private:
  
  /**
   * @brief Generate a series of unique IDs. This method probably does not 
   * need to be static, but it should not hurt either.
   * 
   * @returns A running ID, cycling from 0 to UINT_MAX-1
   */
  static uint get_next_task_ID()
  {
    static uint running_ID(0);
    static boost::mutex LOCK;
    
    LOCK.lock();
    running_ID = (running_ID+1) % UINT_MAX;
    LOCK.unlock();
    
    return running_ID;
  }
  
  
  /**
   * @brief A generator for std::ifstream*. An Ifstream_yielder instance
   * distributes up to max_stream streams via get(). Further streams can only 
   * be retrieved once some streams have been given back to the instance 
   * via put_back(..).
   * 
   * @param max_streams Maximum number of std::ifstream* that can be given out
   */
  struct Ifstream_yielder
  {
    /**
     * @brief Constructor
     * 
     * @param max_streams Upper limit to how many std::ifstream instances this Ifstream_yielder
     */
    Ifstream_yielder(int max_streams)
    : m_reserve(),
      m_max_streams(max_streams)
    {
      for (unsigned int i = 0; i < m_max_streams; ++i) {
        std::ifstream* tmp = new std::ifstream();
        m_reserve.push(tmp);
      }
    }
    
    /**
     * @brief Destructor
     */
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
    
    /**
     * @brief Get a std::ifstream*. This method blocks until a free stream
     * is available
     * 
     * @returns A std::ifstream*
     */
    std::ifstream* get()
    {
      std::ifstream* next_free = m_reserve.pop();
      {
        boost::lock_guard<boost::mutex> LOCK(m_in_use__LOCK);
        m_in_use.push_back(next_free);
      }
      return next_free;
    }
    
    /**
     * @brief Called by users to return std::ifstream* retrieved by get().
     * 
     * @param stream_ptr A std::ifstream* originally given out by this instance
     */
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
  
  /**
   * @brief Use Ifstream_yielder to obtain a managed std::ifstream*.
   * 
   * @param yielder_ptr Ifstream_yielder instance from which the 
   * Ifstream_wrapper will retrieve a std::ifstream*, and to which
   * the stream is given back upon destruction of the Ifstream_wrapper.
   */
  struct Ifstream_wrapper
  {
    /**
     * @brief Constructor
     * 
     * @param yielder_ptr Ifstream_yielder instance from which to get the std::ifstream* stored in m_stream_ptr
     */
    Ifstream_wrapper(Ifstream_yielder* yielder_ptr)
    : m_yielder_ptr(yielder_ptr)
    {
      if (not m_yielder_ptr)
        LOG(FATAL) << "Ifstream_yielder* is invalid";
      m_stream_ptr = m_yielder_ptr->get();
      if (not m_stream_ptr)
        LOG(FATAL) << "std::ifstream* is invalid";
    }
    
    /**
     * @brief Destructor
     */
    ~Ifstream_wrapper() 
    {
      if (not m_yielder_ptr)
        LOG(FATAL) << "Ifstream_yielder* is invalid";
      m_yielder_ptr->put_back(m_stream_ptr);
    }
    
    /**
     * @brief Get this instance's std::ifstream*
     * 
     * @returns m_stream_ptr
     */
    std::ifstream* operator()()
    {
      if (not m_stream_ptr)
        LOG(FATAL) << "std::ifstream* is invalid";
      return m_stream_ptr;
    }
    
    std::ifstream* m_stream_ptr;
    Ifstream_yielder* m_yielder_ptr;
  };
  
  std::vector<Ifstream_yielder*> binstream_yielders_;
  
  /**
   * @brief Model for one image in a binary file. This entry contains the index 
   * of the binary file (to be used with "binfiles_"), the byte offset within 
   * the file at which the relevant data starts (flag + image pixels), and 
   * additionally the data type.
   */
  struct Entry {
    int binfile_idx;
    long int byte_offset;
    BinaryDB_DataEncoding data_encoding;
  };
  
  /**
   * @brief Encapsulate all information and structures needed to read one entry
   */
  struct ReadTask {
    /**
     * @brief Constructor
     * 
     * @param entry_ref Entry to which this read task corresponds
     * @param binstream_yielder_ptr Ifstream_yielder from which to retrieve a std::ifstream
     * @param N Number of bytes to read (4 additional flag bytes will be read)
     * @param dst_ptr Location where the read (and converted) data will be stored
     */
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
    
    /**
     * Debugging stuff
     */
    void debug(int s, int i){ sample=s; index=i; }
    int sample;
    int index;
    uint ID;
  };
  std::queue<ReadTask*> undone_tasks;
  std::vector<uint> in_progress_task_ids;
  std::queue<ReadTask*> done_tasks;
  boost::mutex queues__LOCK;
  std::vector<boost::thread*> worker_threads;
  bool running;

  /**
   * @brief Interpread the first 4 bytes of "buffer" as a uint and panic if the result is not 1
   * @param buffer Data (at least 4 bytes long)
   */
  void check_flag(
        unsigned char* buffer);
  
  /**
   * @brief Called by a worker threads for every ReadTask it processes
   * @param task_ptr The ReadTask to be processed
   * @param entry_buffer_ Thread-local data buffer
   */
  void process_readtask(
        ReadTask* task_ptr,
        unsigned char* entry_buffer_);
  
  /**
   * @brief Entry point for worker threads; Continuously monitor task queues and process entries until "running" is FALSE
   */
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
  /**
   * @brief Split a string at '/' and return the concatenation of the last two elements. Used to extract semi-detailed location from a file path, e.g. "/tmp/my/file/location/file.file" yields "location/file.file"
   * 
   * @param s String (e.g. file path) to be split
   */
  std::string debug_m_split(const std::string& s)
  {
    std::vector<std::string> elems;
    debug_m_split(s, '/', elems);
    std::ostringstream oss;
    oss << elems[elems.size()-2] << '/' << elems.back();
    return oss.str();
  }
  
};


}  // namespace db
}  // namespace caffe

#endif  // CAFFE_UTIL_DB_BINARYDB_HPP
