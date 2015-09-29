#include "caffe/util/binarydb.hpp"

/// System/STL
#include <fstream>
#include <queue>
#include <string>
#include <vector>
/// Boost
#include <boost/chrono.hpp>
#include <boost/python.hpp>
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
/// Caffe/local files
#include "caffe/util/benchmark.hpp"

namespace bp = boost::python;

namespace caffe { 
namespace db {
  
  
  
/**
 * @brief Constructor
 */
template <typename Dtype>
BinaryDB<Dtype>::BinaryDB() {
  num_samples_ = 0;
}


/**
 * @brief Destructor
 */
template <typename Dtype>
BinaryDB<Dtype>::~BinaryDB() 
{ 
  Close();
  
  /// Deinitialize and destroy worker threads
  for (unsigned int i = 0; i < worker_threads.size(); ++i) {
    worker_threads[i]->join();
    delete worker_threads[i];
  }
  
  /// Free buffer memory
  for (unsigned int i = 0; i < entry_buffers_.size(); ++i) {
    if (entry_buffers_[i])
      delete[] entry_buffers_[i];
  }
}


/**
 * @brief Read a configuration file, parse which data can be read and 
 * instantiate internal structures
 * 
 * @param source Configuration file (the file has already been processed
 *               by the Python backend; this parameter is here only used
 *               for debug prints)
 * @param param Caffe parameters
 */
template <typename Dtype>
void BinaryDB<Dtype>::Open(const string& source, const LayerParameter& param) 
{
  top_num_ = param.top_size();
  sample_variants_num_ = param.data_param().sample_size();
  
  LOG(INFO) << "Opening BinaryDB using boost::python";
  
  std::string param_str;
  param.SerializeToString(&param_str);
  
  Py_Initialize();
  try {
    bp::object module = bp::import("binarydb");
    bp::object dbclass = module.attr("BinaryDB")(param_str, top_num_);
    
    // returns (all_samples, entry_dimensions, bin_filenames)
    bp::tuple infos = (bp::tuple)dbclass.attr("getInfos")();
    
    if(bp::len(infos) != 3) 
      LOG(FATAL) << "Python did not return 3-tuple";
    
    bp::list all_samples   = (bp::list)infos[0];
    bp::list dimensions    = (bp::list)infos[1];
    bp::list bin_filenames = (bp::list)infos[2];
    
    // Store dimensions:
    if(bp::len(dimensions) != top_num_) 
      LOG(FATAL) << "Number of entry dimensions passed from python"
                 << " not equal to top blob count";
    
    entry_dimensions_.resize(top_num_);
    entry_buffer_size_ = 0;
    for(int entry = 0; entry < top_num_; entry++) {
      entry_dimensions_[entry].resize(4);
      entry_dimensions_[entry][0] = 1;
      entry_dimensions_[entry][1] = bp::extract<int>(dimensions[entry][2]);
      entry_dimensions_[entry][2] = bp::extract<int>(dimensions[entry][1]);
      entry_dimensions_[entry][3] = bp::extract<int>(dimensions[entry][0]);

      int size = sizeof(float);
      for(int i=0; i<4; i++)
          size*=entry_dimensions_[entry][i];

      if(size>entry_buffer_size_)
          entry_buffer_size_ = size;
    }
    
    entry_buffers_.resize(top_num_);
    for(unsigned int i = 0; i < entry_buffers_.size(); ++i) {
      /// Expected maximum read size + flag
      entry_buffers_[i] = new unsigned char[entry_buffer_size_+4];
    }
    
    // Store bin filenames:
    binfiles_.resize(bp::len(bin_filenames));
    for(int i=0; i<bp::len(bin_filenames); i++) {
      binfiles_[i] = bp::extract<string>(bin_filenames[i]);
    }
    
    // Store samples
    // [[(0, 3110400, 1), (1, 4147200, 3), (2, 3110400, 3)], [(),(),()], ... ]
    num_samples_ = bp::len(all_samples);
    samples_.resize(num_samples_);
    for(int sample=0; sample<num_samples_; sample++) {
      samples_[sample].resize(top_num_);
      for(int entry=0; entry<top_num_; entry++) {
        samples_[sample][entry].binfile_idx = bp::extract<int>(all_samples[sample][entry][0]);
        samples_[sample][entry].byte_offset = bp::extract<long int>(all_samples[sample][entry][1]);
        int encoding_int = bp::extract<int>(all_samples[sample][entry][2]);
        samples_[sample][entry].data_encoding = (BinaryDB_DataEncoding)encoding_int;
      }
    }
    
  } catch (bp::error_already_set) {
    PyErr_Print();
    throw;
  }
  
  // open binfiles
//   binstreams_.resize(binfiles_.size());
//   for (int i = 0; i < binfiles_.size(); ++i) {
//     binstreams_.at(i).reset(new std::ifstream(binfiles_.at(i).c_str(), 
//                                               std::ios::in | std::ios::binary));
//     if(!binstreams_.at(i)->is_open() or !binstreams_.at(i)->good()) {
//       LOG(FATAL) << "Could not open bin file " << binfiles_.at(i);
//     }
//   }
  /// Open a file stream for every top blob
  binstreams_.resize(top_num_);
  for (unsigned int i = 0; i < top_num_; ++i) {
    int file_idx = samples_[0][i].binfile_idx;
    binstreams_.at(i).reset(new std::ifstream(binfiles_.at(file_idx).c_str(), 
                                              std::ios::in | std::ios::binary));
    if(!binstreams_.at(i)->is_open() or !binstreams_.at(i)->good()) {
      LOG(FATAL) << "Could not open bin file " << binfiles_.at(file_idx);
    }
  }
  
  /// Create and start worker threads (one for each ENTRY)
  running = true;
  worker_threads.resize(top_num_);
  for (unsigned int i=0; i < worker_threads.size(); ++i) {
    worker_threads[i] = new boost::thread(&BinaryDB<Dtype>::worker_thread_loop,
                                          this);
  }

  
  // permute the samples
  if (param.data_param().rand_permute()) {  
    int seed = param.data_param().rand_permute_seed();
    if(seed > 0) std::srand (unsigned(seed));
    std::random_shuffle ( samples_.begin(), samples_.end() );  
  }
  
  LOG(INFO) << "Opened BinaryDB " << source;
}


/**
 * @brief Tidy up and close std::ifstream instances
 * 
 * @note Called by destructor
 */
template <typename Dtype>
void BinaryDB<Dtype>::Close() 
{
  running = false;
  
  /// Close file handles
  for (unsigned int i = 0; i < binstreams_.size(); ++i)
    binstreams_.at(i)->close();
}


/**
 * @brief 
 * 
 * @param index 
 * @param dst 
 * @param compressed_size 
 */
template <typename Dtype>
void BinaryDB<Dtype>::get_sample(int index, 
                                 vector<Blob<Dtype>*>* dst, 
                                 int* compressed_size) 
{
  if(compressed_size) *compressed_size=0;
  
  /// Create a read task for every top blob
  undone_tasks__LOCK.lock();
  for (unsigned int t = 0; t < top_num_; ++t) {
    dst->at(t)->Reshape(entry_dimensions_.at(t));
    ReadTask* new_task_ptr = new ReadTask(samples_.at(index).at(t),
                                          binstreams_.at(t),
                                          dst->at(t)->count(),
                                          dst->at(t)->mutable_cpu_data(),
                                          entry_buffers_.at(t));
    undone_tasks.push(new_task_ptr);
  }
  undone_tasks__LOCK.unlock();
    
//     Entry entry = samples_.at(index).at(t);    
//     std::ifstream* curr_binstream = binstreams_.at(entry.binfile_idx).get();
// 
//     Timer timer;
//     timer.Start();
//     curr_binstream->seekg(entry.byte_offset, ios::beg);
//     
//     // check if the stream is ok, re-open if needed
//     if (!curr_binstream->is_open() or !curr_binstream->good()) {
//       LOG(INFO) << "Something is wrong with the stream of file " 
//                 << binfiles_.at(entry.binfile_idx);
//       
//       LOG(INFO) << " is_open()=" << curr_binstream->is_open();
//       LOG(INFO) << "    good()=" << curr_binstream->good();
//       LOG(INFO) << "     eof()=" << curr_binstream->eof();
//       LOG(INFO) << "    fail()=" << curr_binstream->fail();
//       LOG(INFO) << "     bad()=" << curr_binstream->bad();
//       
//       LOG(INFO) << "Attempting to re-open";
//       binstreams_.at(entry.binfile_idx).get()->close();
//       binstreams_.at(entry.binfile_idx).get()->open(
//                               binfiles_.at(entry.binfile_idx).c_str(),
//                               std::ios::in | std::ios::binary);
//       
//       if(!curr_binstream->is_open() or !curr_binstream->good())
//         LOG(FATAL) << "Could not re-open bin file " 
//                    << binfiles_.at(entry.binfile_idx);
//       else {
//         curr_binstream = binstreams_.at(entry.binfile_idx).get();
//         curr_binstream->seekg(entry.byte_offset, ios::beg);
//       }
//     }
//     
//     // check the flag of the entry (4-byte thing)
//     unsigned int flag;
//     curr_binstream->read(reinterpret_cast<char *>(&flag), 4);
//     if (flag == 0)
//       LOG(FATAL) << "Flag of blob " << t << " of sample " << index 
//                  << " is 0 " << " (File " << binfiles_.at(entry.binfile_idx)
//                  << " offset " << entry.byte_offset << ")";
//     else if (flag != 1)
//       LOG(FATAL) << "Flag of blob " << t << " of sample " << index 
//                  << " has invalid value " << flag << " (File " 
//                  << binfiles_.at(entry.binfile_idx) << " offset " 
//                  << entry.byte_offset << ")";
//     
//     timer.Stop();
//     TimingMonitor::addMeasure("seek_time", timer.MilliSeconds());
// 
//     // reshape blob
//     dst->at(t)->Reshape(entry_dimensions_[t]);
//     
//     // actually read the data
// 
//     int n_read;
//     read_binstream(curr_binstream, 
//                    entry.data_encoding, 
//                    dst->at(t)->count(), 
//                    dst->at(t)->mutable_cpu_data(), 
//                    &n_read);
//     if(compressed_size) *compressed_size+=n_read;
//   }
  
  /// Wait until all read tasks are complete
  while (done_tasks.size() < top_num_) {
    boost::this_thread::sleep_for(boost::chrono::milliseconds(10));
  }
  /// Delete done tasks
  done_tasks__LOCK.lock();
  while (done_tasks.size() > 0) {
    ReadTask* task_ptr = done_tasks.front();
    if(compressed_size) 
      *compressed_size += task_ptr->n_read;
    done_tasks.pop();
    delete task_ptr;
  }
  done_tasks__LOCK.unlock();
}


/**
 * @brief Given a ifstream (with file pointer set correctly), reads N values 
 * from the stream to out with the given data_encoding
 * 
 * @param binstream 
 * @param data_encoding 
 * @param N 
 * @param out 
 * @param n_read 
 */
// template <typename Dtype>
// void BinaryDB<Dtype>::read_binstream(std::ifstream* binstream, 
//                                      BinaryDB_DataEncoding data_encoding, 
//                                      long int N, 
//                                      Dtype* out, 
//                                      int* n_read)
// {
//   
// }


/**
 * @brief Check the first 4 bytes of "buffer"
 */
template <typename Dtype>
void BinaryDB<Dtype>::check_flag(unsigned char* buffer)
{
  uint flag = (reinterpret_cast<uint*>(buffer))[0];
  if (flag != 1)
    LOG(FATAL) << "Bad flag: " << flag;
//   if (flag == 0)
//     LOG(FATAL) << "Flag of blob " << t << " of sample " << index 
//                << " is 0 " << " (File " << binfiles_.at(entry.binfile_idx)
//                << " offset " << entry.byte_offset << ")";
//   else if (flag != 1)
//     LOG(FATAL) << "Flag of blob " << t << " of sample " << index 
//                << " has invalid value " << flag << " (File " 
//                << binfiles_.at(entry.binfile_idx) << " offset " 
//                << entry.byte_offset << ")";
}


/**
 * Process a ReadTask instance (used by worker threads)
 * 
 * @param task_ptr ReadTask instance to process
 */
template <typename Dtype>
void BinaryDB<Dtype>::process_readtask(ReadTask* task_ptr)
{  
//   read_binstream( task_ptr->entry_ref,
//                   task_ptr->binstream_ptr.get(),
//                   task_ptr->entry_ref.data_encoding,
//                   task_ptr->N,
//                   task_ptr->dst_ptr,
//                   &(task_ptr->n_read) );
  
  task_ptr->n_read=0;
  
  Entry& entry = task_ptr->entry_ref; 
  std::ifstream* binstream = task_ptr->binstream_ptr.get();
  
  Dtype* out = task_ptr->dst_ptr;

  Timer timer;
  timer.Start();
  binstream->seekg(entry.byte_offset, ios::beg);
  
  // check if the stream is ok, re-open if needed
  if (!binstream->is_open() or !binstream->good()) {
    LOG(INFO) << "Something is wrong with the stream of file " 
              << binfiles_.at(entry.binfile_idx);
    
    LOG(INFO) << " is_open()=" << binstream->is_open();
    LOG(INFO) << "    good()=" << binstream->good();
    LOG(INFO) << "     eof()=" << binstream->eof();
    LOG(INFO) << "    fail()=" << binstream->fail();
    LOG(INFO) << "     bad()=" << binstream->bad();
    
    LOG(INFO) << "Attempting to re-open";
    binstream->close();
    binstream->open(binfiles_.at(entry.binfile_idx).c_str(),
                    std::ios::in | std::ios::binary);
    
    if(!binstream->is_open() or !binstream->good())
      LOG(FATAL) << "Could not re-open bin file " 
                  << binfiles_.at(entry.binfile_idx);
    else {
      //binstream = binstreams_.at(entry.binfile_idx).get();
      binstream->seekg(entry.byte_offset, ios::beg);
    }
  }
    
  timer.Stop();
  TimingMonitor::addMeasure("seek_time", timer.MilliSeconds());

  // reshape blob
//   dst->at(t)->Reshape(entry_dimensions_[t]);

  Timer t1;
  Timer t2;
  
  uint N = task_ptr->N;
  
  unsigned char* entry_buffer_ = task_ptr->entry_buffer;

  switch(entry.data_encoding)
  {
    case BinaryDB_DataEncoding_UINT8:
      if(entry_buffer_size_<N)
        LOG(FATAL) << "UINT8: entry buffer too small, buffer size=" 
                   << entry_buffer_size_ << ", N=" << N;

      t1.Start(); binstream->read((char*)entry_buffer_, N+4); t1.Stop();
//       check_flag(entry_buffer_);
      entry_buffer_ += 4;
      
      TimingMonitor::addMeasure("raw_data_rate", N * 1000.0 / 
                                (t1.MilliSeconds() * 1024.0 * 1024.0));
      task_ptr->n_read += N;

      t2.Start();
      for(int i=0; i<N; i++)  *(out++)=static_cast<Dtype>(entry_buffer_[i]);
      t2.Stop();
      TimingMonitor::addMeasure("decomp_data_rate", N * 1000.0 / 
                                (t2.MilliSeconds() * 1024.0 * 1024.0));
      break;

    case BinaryDB_DataEncoding_FIXED16DIV32:
      if(entry_buffer_size_<2*N)
        LOG(FATAL) << "FIXED16DIV32: entry buffer too small, buffer size=" 
                   << entry_buffer_size_ << ", 2*N=" << 2*N;

      t1.Start(); binstream->read((char*)entry_buffer_, 2*N+4); t1.Stop();
//       check_flag(entry_buffer_);
      entry_buffer_ += 4;
      
      TimingMonitor::addMeasure("raw_data_rate", 2*N * 1000.0 / 
                                (t1.MilliSeconds() * 1024.0 * 1024.0));
      task_ptr->n_read += 2*N;

      t2.Start();
      for(int i=0; i<N; i++) {
        short v = *((short*)(&entry_buffer_[2*i]));

        Dtype value;
        if(v==std::numeric_limits<short>::max()) 
          value = std::numeric_limits<Dtype>::signaling_NaN();
        else
          value = ((Dtype)v)/32.0;

        *(out++)=value;
      }
      t2.Stop();
      TimingMonitor::addMeasure("decomp_data_rate", 2*N * 1000.0/ 
                                (t2.MilliSeconds() * 1024.0 * 1024.0));
      break;

    default:
        LOG(FATAL) << "Unknown data encoding " << entry.data_encoding;
        break;
  }
}


/**
 * Entry point for worker threads
 */
template <typename Dtype>
void BinaryDB<Dtype>::worker_thread_loop()
{
  while (running) 
  {
    if (undone_tasks.size() > 0)
    {
      /// Fetch task
      undone_tasks__LOCK.lock();
      ReadTask* task_ptr = undone_tasks.front();
//       LOG(INFO) << "Thread (" << boost::this_thread::get_id() << ") fetched"
//                 << " a new task.";
      undone_tasks.pop();
      undone_tasks__LOCK.unlock();
      
      /// Process task
      process_readtask(task_ptr);
      
      /// Mark task as done
      done_tasks__LOCK.lock();
      done_tasks.push(task_ptr);
      done_tasks__LOCK.unlock();
    }
    else
    {
      boost::this_thread::sleep_for(boost::chrono::milliseconds(10));
    }
  }
}



INSTANTIATE_CLASS(BinaryDB);


}  // namespace db
}  // namespace caffe

