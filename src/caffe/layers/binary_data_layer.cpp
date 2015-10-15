// Copyright 2014 BVLC and contributors.

/// System/STL
#include <stdint.h>
#include <pthread.h>
#include <vector>

/// Boost
#include <boost/thread.hpp>

/// Caffe and local files
#include "caffe/binary_data_reader.hpp"
#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"
#include <boost/function.hpp>

namespace caffe {


#define Container vector<Blob<Dtype>*>


/**
 * @brief Constructor
 * 
 * @param param Parameters
 */
template <typename Dtype>
BinaryDataLayer<Dtype>::BinaryDataLayer(const LayerParameter& param)
  : Layer<Dtype>(param),
    prefetch_(),
    prefetch_free_(),
    prefetch_full_(),
    reader_(param)
{
  prefetch_.resize(param.data_param().prefetch()*
                   param.data_param().batch_size());

  CHECK_GE(param.data_param().sample().size(), 0) << "No samples defined.";  

  output_index_ = false;
  if(param.top_size() == param.data_param().sample().Get(0).entry_size() + 1)
      output_index_ = true;

  /// Populate prefetching queue with empty buckets
  for (int i = 0; i < prefetch_.size(); ++i) {
    for (int j = 0; j < param.top_size(); ++j) {
      Blob<Dtype> *tmpblob = new Blob<Dtype>();
      prefetch_[i].push_back(tmpblob);
    }
    prefetch_free_.push(&prefetch_[i]);
  }
}


/**
 * @brief Destructor
 */
template <typename Dtype>
BinaryDataLayer<Dtype>::~BinaryDataLayer<Dtype>() {
  this->StopInternalThread();

  /// Tidy up
  /// Depublish data buckets and ensure that all are accounted for
  {
    unsigned int total = prefetch_.size();
    while (prefetch_free_.size() > 0) {
      prefetch_free_.pop();
      --total;
    }
    while (prefetch_full_.size() > 0) {
      prefetch_full_.pop();
      --total;
    }
    if (total > 0)
      LOG(INFO) << "Warning: There are " << prefetch_.size() << " allocated"
                << " prefetching buckets, but " << total << " of these could"
                << " not be accounted for during cleanup. This is not fatal,"
                << " but might indicate a memory leak.";
    else if (total < 0)
      LOG(INFO) << "Warning: There are " << prefetch_.size() << " allocated"
                << " prefetching buckets, but " << -total+prefetch_.size() 
                << " were found in the prefetching queues during cleanup."
                << " This is not fatal, but probably bad.";
  }
  
  /// Free raw Blob memory within the buckets
  for (unsigned int i = 0; i < prefetch_.size(); ++i) {
    Container& container = prefetch_[i];
    for (unsigned int j = 0; j < container.size(); ++j) {      if (container[j])
        delete container[j];
    }
  }
}


/**
 * @brief Initial layer setup
 */
template <typename Dtype>
void BinaryDataLayer<Dtype>::LayerSetUp(const Container& bottom,
                                        const Container& top)
{
  const bool verbose = this->layer_param_.data_param().verbose();
  if (verbose && this->layer_param_.data_param().block_size())
    LOG(INFO) << "Block size: " << this->layer_param_.data_param().block_size();

  /// Look at a data sample and use it to initialize the top blobs
  const int batch_size = this->layer_param_.data_param().batch_size();
  Container& peek_data = *(reader_.full().peek());
  if(!output_index_)
  {
      // output only blobs
      assert(top.size() == peek_data.size());
      for (unsigned int i = 0; i < top.size(); ++i)
      {
        vector<int> shape(peek_data[i]->shape());
        shape[0] = batch_size;
        top[i]->Reshape(shape);
      }
  }
  else
  {
      assert((top.size() + 1 == peek_data.size()));
      // Output blobs and sample index
      for (unsigned int i = 0; i < top.size()-1; ++i)
      {
        vector<int> shape(peek_data[i]->shape());
        shape[0] = batch_size;
        top[i]->Reshape(shape);
      }
      top.back()->Reshape(this->layer_param_.data_param().batch_size(), 1, 1, 1);
  }
  
  DLOG(INFO) << "Initializing prefetch";
  StartInternalThread();
  DLOG(INFO) << "Prefetch initialized.";

  if(this->phase_ == TEST)
  {
      if (reader_.get_num_samples() % batch_size)
          LOG(FATAL) << "Number of samples (" << reader_.get_num_samples() << ") of TEST set is not divisible by batch size (" << batch_size << ")";

      this->GetNet()->set_test_iter_count(reader_.get_num_samples()/batch_size);
  }

  if(this->layer_param().data_param().error_based_sampling())
  {
      typename Net<Dtype>::UpdateSampleCallback cb;

      cb = boost::bind(&BinaryDataReader<Dtype>::update_sample_errors, &reader_, _1, _2);

      this->GetNet()->register_update_sample_callback(cb);
  }
}

/**
 * Entry point for InternalThread (not called by user)
 * 
 * Fetches an empty bucket from the prefetch_free_ queue, fills it, and
 * pushes it into the prefetch_full_ queue.
 */
template <typename Dtype>
void BinaryDataLayer<Dtype>::InternalThreadEntry() 
{
#ifndef CPU_ONLY
  cudaStream_t stream;
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
#endif

  try {
    while (!must_stop()) {
      Container* container_ptr = prefetch_free_.pop();
      /// Fetch data from reader; stall while no data is available
      load_batch(container_ptr);
#ifndef CPU_ONLY
      if (Caffe::mode() == Caffe::GPU) {
        for (unsigned int i = 0; i < container_ptr->size(); ++i)
          (*container_ptr)[i]->data().get()->async_gpu_push(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }
#endif
      prefetch_full_.push(container_ptr);
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
#endif
}


/**
 * Fetch a data batch from the internal reader (not called by user)
 * 
 * @param output_ptr Prefetch target into which data is copied. The prefetch
 *                   target is a batch; each Blob in the vector stores multiple
 *                   samples
 */
template <typename Dtype>
void BinaryDataLayer<Dtype>::load_batch(Container* output_ptr)
{
  const Container& output = (*output_ptr);
  
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;

  /// Reshape output to match source data
  const int batch_size = this->layer_param_.data_param().batch_size();
  Container& data = *(reader_.full().peek());
  assert(output.size() == data.size());
  for (unsigned int i = 0; i < output.size(); ++i) 
  {
    vector<int> shape(data[i]->shape());
    shape[0] = batch_size;
    output[i]->Reshape(shape);
  }
  
  for (unsigned int i = 0; i < output.size(); ++i)
    CHECK(output[i]->count());

  /// Fill output
  for (unsigned int i = 0; i < batch_size; ++i)
  {
    timer.Start();
    /// Fetch one data sample from internal reader
    Container& data = *(reader_.full().pop("Waiting for data"));
    read_time += timer.MicroSeconds();
    timer.Start();
    /// Copy data from new sample into output
    for (unsigned int j = 0; j < output.size(); ++j)
    {
      Blob<Dtype>* target_ptr = output[j];
      Blob<Dtype>* const source_ptr = data[j];
      const int offset = target_ptr->offset(i, 0, 0, 0);
      caffe_copy(source_ptr->count(),
                 source_ptr->cpu_data(),
                 target_ptr->mutable_cpu_data()+offset);
    }
    trans_time += timer.MicroSeconds();
    
    /// Recycle spent data container for data reading
    reader_.free().push(const_cast<Container*>(&data));
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}



/**
 * @brief Forward_cpu
 * 
 * Fetches a full bucket from the prefetch_full_ queue, uses its contents,
 * and pushes it back into the prefetch_free_ queue.
 */
template <typename Dtype>
void BinaryDataLayer<Dtype>::Forward_cpu(const Container& bottom,
                                         const Container& top) 
{  
  TimingMonitor::addMeasure("batch_size", this->layer_param_.data_param().batch_size());

  reentry_timer_.Stop();
  TimingMonitor::addMeasure("train_reentry", reentry_timer_.MilliSeconds());

  Timer wait_timer;
  wait_timer.Start();
  Container* container_ptr = prefetch_full_.pop("Data layer prefetch queue empty");
  const Container& container = (*container_ptr);
  wait_timer.Stop();
  TimingMonitor::addMeasure("train_wait", wait_timer.MilliSeconds());

  /// Reshape tops and copy data
  for (unsigned int i = 0; i < top.size(); ++i) {
    top[i]->ReshapeLike(*container[i]);
    caffe_copy(container[i]->count(), 
               container[i]->cpu_data(),  
               top[i]->mutable_cpu_data());
    
  }

  /// Recycle spent data container for prefetching
  prefetch_free_.push(container_ptr);

  reentry_timer_.Start();
}


template <typename Dtype>
void BinaryDataLayer<Dtype>::Forward_gpu(const Container& bottom,
                                         const Container& top) 
{
  /// TODO
  Forward_cpu(bottom, top);
}


#ifdef CPU_ONLY
STUB_GPU_FORWARD(BinaryDataLayer, Forward);
#endif

INSTANTIATE_CLASS(BinaryDataLayer);
REGISTER_LAYER_CLASS(BinaryData);

}  // namespace caffe
