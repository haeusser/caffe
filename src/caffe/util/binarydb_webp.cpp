
#include "caffe/util/binarydb_webp.hpp"

/// System/STL
#include <cstring>
#include <fstream>
#include <queue>
#include <string>
#include <vector>
/// Boost
#include <boost/chrono.hpp>
#include <boost/python.hpp>
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
/// WebP (local)
#include <webp/decode.h>
/// Caffe/local files
#include "caffe/util/benchmark.hpp"
#include "caffe/util/lzo/decompress.hpp"

namespace bp = boost::python;

namespace caffe {
namespace db {



/**
 * @brief Decompress a WebP-encoded image
 * @param in_out Compressed input data/decompressed output data (must be large enough to hold the decompressed image)
 * @param compressed_size Number of data bytes in compressed blob
 */
void decodeWebP(unsigned char* in_out,
                unsigned int compressed_size)
{
  int width, height;
  uint8_t* decoded_data = WebPDecodeRGB(reinterpret_cast<uint8_t*>(in_out),
                                        compressed_size,
                                        &width,
                                        &height);
  /// Reorder bytes (rgbrgbrgbrgb)->(rrrrggggbbbb)
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < 3; ++c) {
        in_out[width*height*c + width*y + x] = decoded_data[(y*width+x)*3+(2-c)];
      }
    }
  }
  free(decoded_data);
}

  

template <typename Dtype>
BinaryDBWebP<Dtype>::BinaryDBWebP() {
  num_samples_ = 0;
}



template <typename Dtype>
BinaryDBWebP<Dtype>::~BinaryDBWebP() 
{ 
  Close();
  
  /// Destroy worker threads
  for (unsigned int i = 0; i < worker_threads.size(); ++i) {
    delete worker_threads[i];
  }
  
  /// Destroy filestreams
  for (unsigned file_idx = 0; file_idx < binfiles_.size(); ++file_idx)
    if (binstream_yielders_[file_idx])
      delete binstream_yielders_[file_idx];
}


template <typename Dtype>
void BinaryDBWebP<Dtype>::Open(const string& source, const LayerParameter& param) 
{
  top_num_ = param.top_size();
  sample_variants_num_ = param.data_param().sample_size();
  top_num_ = param.data_param().sample().Get(0).entry_size();

  LOG(INFO) << "Opening BinaryDBWebP using boost::python";
  
  std::string param_str;
  param.SerializeToString(&param_str);
  
  Py_Initialize();
  try {
    bp::object module = bp::import("binarydb_webp");
    bp::object dbclass = module.attr("BinaryDBWebP")(param_str, top_num_);
    
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
    entry_buffer_size_ += 16;
    
    
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
        samples_[sample][entry].binfile_idx = 
            bp::extract<int>(all_samples[sample][entry][0]);
        samples_[sample][entry].byte_offset = 
            bp::extract<long int>(all_samples[sample][entry][1]);
        int encoding_int = 
            bp::extract<int>(all_samples[sample][entry][2]);
        samples_[sample][entry].data_encoding =
            (BinaryDBWebP_DataEncoding)encoding_int;
        samples_[sample][entry].compressed_byte_size =
            bp::extract<int>(all_samples[sample][entry][3]);
      }
    }
    
  } catch (bp::error_already_set) {
    PyErr_Print();
    throw;
  }
  
  /// Open a file stream for every combination of (accessed file X offset)
//     int file_idx = samples_[0][i].binfile_idx;
  binstream_yielders_.resize(binfiles_.size());
  for (unsigned i = 0; i < binfiles_.size(); ++i) {
    binstream_yielders_[i] = new Ifstream_yielder(3);
  }
  

  // permute the samples
  if (param.data_param().rand_permute()) {
    int seed = param.data_param().rand_permute_seed();
    if(seed > 0) std::srand (unsigned(seed));
    std::random_shuffle(samples_.begin(), samples_.end());
  }

  
  /// Create and start worker threads (one for each ENTRY)
  running = true;
  if (param.data_param().disk_reader_threads() > 0)
    worker_threads.resize(param.data_param().disk_reader_threads());
  else
    worker_threads.resize(top_num_);
  LOG(INFO) << "Spawning " << worker_threads.size() << " worker threads.";
  for (unsigned int i=0; i < worker_threads.size(); ++i) {
    worker_threads[i] = new boost::thread(&BinaryDBWebP<Dtype>::worker_thread_loop,
                                          this);
  }
  
  LOG(INFO) << "Opened BinaryDBWebP " << source;
}


template <typename Dtype>
void BinaryDBWebP<Dtype>::Close() 
{
  running = false;
  
  /// Stop worker threads
  for (unsigned int i = 0; i < worker_threads.size(); ++i) {
    worker_threads[i]->join();
  }
}


template <typename Dtype>
void BinaryDBWebP<Dtype>::get_sample(int index, 
                                 vector<Blob<Dtype>*>* dst, 
                                 int* compressed_size,
                                 bool wait_for_finish) 
{
  if (not dst)
    LOG(FATAL) << ">dst< target blob vector is NULL";
  
  /// Create a read task for every top blob
  {
    boost::lock_guard<boost::mutex> LOCK(queues__LOCK);
    for (unsigned int t = 0; t < top_num_; ++t) {
      dst->at(t)->Reshape(entry_dimensions_.at(t));
      if (not binstream_yielders_.at(samples_.at(index).at(t).binfile_idx))
        LOG(FATAL) << "Ifstream_yielder " << samples_.at(index).at(t).binfile_idx 
                    << " is invalid";
                    
      ReadTask* new_task_ptr = new ReadTask(
                        samples_.at(index).at(t),
                        binstream_yielders_.at(samples_.at(index).at(t).binfile_idx),
                        dst->at(t)->count(),
                        dst->at(t)->mutable_cpu_data());
      new_task_ptr->debug(index, t);
      undone_tasks.push(new_task_ptr);
    }
    if(dst->size()>top_num_)
    {
        dst->back()->Reshape(1,1,1,1);
        dst->back()->mutable_cpu_data()[0] = index;
    }
  }
  
  if (wait_for_finish) {
    /// Wait until all read tasks are complete
    if (compressed_size) *compressed_size = 0;
    bool all_done = false;
    while (not all_done) {
      if (undone_tasks.size() > 0 or in_progress_task_ids.size() > 0) {
        boost::this_thread::sleep_for(boost::chrono::milliseconds(10));
      } else {
        all_done = true;
      }
    }
    /// Delete done tasks
    while (done_tasks.size() > 0) {
      boost::lock_guard<boost::mutex> LOCK(queues__LOCK);
      ReadTask* task_ptr = done_tasks.front();
      if (compressed_size) *compressed_size += task_ptr->n_read;
      delete task_ptr;
      done_tasks.pop();
    }
  }
}


template <typename Dtype>
void BinaryDBWebP<Dtype>::check_flag(unsigned char* buffer)
{
  if (not buffer)
    LOG(FATAL) << "Bad buffer";
    
  uint flag = (reinterpret_cast<uint*>(buffer))[0];
  if (flag != 1)
    LOG(FATAL) << "Bad flag: " << flag;
}


template <typename Dtype>
void BinaryDBWebP<Dtype>::process_readtask(ReadTask* task_ptr,
                                           unsigned char* entry_buffer_,
                                           lzo::LZO_Decompressor* lzo_decompressor)
{
  task_ptr->n_read=0;
  
  Entry& entry = task_ptr->entry_ref; 
  if (not task_ptr->binstream_yielder_ptr)
    LOG(FATAL) << "Ifstream_yielder is invalid";  
  Ifstream_wrapper binstream_wrapper(task_ptr->binstream_yielder_ptr);
  std::ifstream* binstream = binstream_wrapper();
  
  if (not binstream->is_open())
    binstream->open(binfiles_.at(entry.binfile_idx).c_str(),
                    std::ios::in | std::ios::binary);
  
  Dtype* out = task_ptr->dst_ptr;

  Timer timer;
  timer.Start();
  if (binstream->eof()) {
    LOG(INFO) << "Clearing EOFBIT for file " << binfiles_.at(entry.binfile_idx);
    binstream->clear();
  }
  if (binstream->eof())
    binstream->clear();
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
      binstream->clear();
      binstream->seekg(entry.byte_offset, ios::beg);
    }
  }
    
  timer.Stop();
  TimingMonitor::addMeasure("seek_time", timer.MilliSeconds());

  Timer t1;
  Timer t2;
  
  uint N = task_ptr->N;
  
  switch(entry.data_encoding)
  {
    case BinaryDBWebP_DataEncoding_UINT8:
      if(entry_buffer_size_<(N+4))
        LOG(FATAL) << "UINT8: entry buffer too small, buffer size=" 
                   << entry_buffer_size_ << ", N+4=" << N+4;

      t1.Start(); binstream->read((char*)entry_buffer_, N+4); t1.Stop();
      
      if (!binstream->is_open() or !binstream->good()) {
        LOG(INFO) << "! EOF ! " << binfiles_.at(task_ptr->entry_ref.binfile_idx);
      }
      
      check_flag(entry_buffer_);
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
      
    case BinaryDBWebP_DataEncoding_UINT8WEBP:
      if(entry_buffer_size_<(N+4))
        LOG(FATAL) << "UINT8: entry buffer too small, buffer size=" 
                   << entry_buffer_size_ << ", N+4=" << N+4;

      t1.Start(); 
      binstream->read((char*)entry_buffer_, entry.compressed_byte_size);
      t1.Stop();
      
      if (!binstream->is_open() or !binstream->good()) {
        LOG(INFO) << "! EOF ! " << binfiles_.at(task_ptr->entry_ref.binfile_idx);
      }
      
      check_flag(entry_buffer_);
      entry_buffer_ += 4;
      
      decodeWebP(entry_buffer_, entry.compressed_byte_size-4);
      
      TimingMonitor::addMeasure("raw_data_rate", N * 1000.0 / 
                                (t1.MilliSeconds() * 1024.0 * 1024.0));
      task_ptr->n_read += N;

      t2.Start();
      for(int i=0; i<N; i++)  *(out++)=static_cast<Dtype>(entry_buffer_[i]);
      t2.Stop();
      TimingMonitor::addMeasure("decomp_data_rate", N * 1000.0 / 
                                (t2.MilliSeconds() * 1024.0 * 1024.0));
      break;

    case BinaryDBWebP_DataEncoding_FIXED16DIV32:
      if(entry_buffer_size_<(2*N+4))
        LOG(FATAL) << "FIXED16DIV32: entry buffer too small, buffer size=" 
                   << entry_buffer_size_ << ", 2*N+4=" << 2*N+4;

      t1.Start(); binstream->read((char*)entry_buffer_, 2*N+4); t1.Stop();
      
      if (!binstream->is_open() or !binstream->good()) {
        LOG(INFO) << "! EOF ! " << binfiles_.at(task_ptr->entry_ref.binfile_idx);
      }

      
      check_flag(entry_buffer_);
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
      
    case BinaryDBWebP_DataEncoding_FIXED16DIV32LZO:
      if(entry_buffer_size_<(2*N+4))
        LOG(FATAL) << "FIXED16DIV32: entry buffer too small, buffer size=" 
                   << entry_buffer_size_ << ", 2*N+4=" << 2*N+4;

      t1.Start(); 
      binstream->read((char*)entry_buffer_, entry.compressed_byte_size); 
      t1.Stop();
      
      if (!binstream->is_open() or !binstream->good()) {
        LOG(INFO) << "! EOF ! " << binfiles_.at(task_ptr->entry_ref.binfile_idx);
      }

      check_flag(entry_buffer_);
      entry_buffer_ += 4;
      
      lzo_decompressor->Decompress(entry_buffer_, entry.compressed_byte_size-4);
      
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



template <typename Dtype>
void BinaryDBWebP<Dtype>::worker_thread_loop()
{
  /// Thread-local buffer
  unsigned char* this_thread_entry_buffer = new unsigned char[entry_buffer_size_];
  /// Thread-local LZO decompressor instance
  lzo::LZO_Decompressor* this_thread_lzo_decompressor = 
      new lzo::LZO_Decompressor(entry_buffer_size_);
  
  while (running) 
  {
    if (undone_tasks.size() > 0)
    {
      /// Fetch task
      ReadTask* task_ptr;
      {
        boost::lock_guard<boost::mutex> LOCK(queues__LOCK);
        if (undone_tasks.size() == 0)
          continue;
        
        task_ptr = undone_tasks.front();
        in_progress_task_ids.push_back(task_ptr->ID);
        undone_tasks.pop();
      }
      
      /// Process task
      process_readtask(task_ptr, 
                       this_thread_entry_buffer, 
                       this_thread_lzo_decompressor);
      
      /// Mark task as done
      {
        boost::lock_guard<boost::mutex> LOCK(queues__LOCK);
        done_tasks.push(task_ptr);
        /// http://stackoverflow.com/a/3385251
        in_progress_task_ids.erase(std::remove(in_progress_task_ids.begin(),
                                              in_progress_task_ids.end(),
                                              task_ptr->ID),
                                  in_progress_task_ids.end());
      }
    }
    else
    {
      boost::this_thread::sleep_for(boost::chrono::milliseconds(10));
    }
  }
  
  if (this_thread_entry_buffer)
    delete[] this_thread_entry_buffer;
  if (this_thread_lzo_decompressor)
    delete this_thread_lzo_decompressor;
}



INSTANTIATE_CLASS(BinaryDBWebP);


}  // namespace db
}  // namespace caffe

