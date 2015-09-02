// Copyright 2014 BVLC and contributors.

/// System/STL
#include <cstdint>
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
    prefetch_free_(),
    prefetch_full_(),
    reader_(param)
{
  /// Populate prefetching queue with empty buckets
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
    prefetch_free_.push(&prefetch_[i]);
  }
}


/**
 * @brief Destructor
 */
template <typename Dtype>
BinaryDataLayer<Dtype>::~BinaryDataLayer<Dtype>() {
  this->StopInternalThread();
  /// TODO Clean up the reader?
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
  Container* peek_data_ptr = reader_.full().peek();
  const Container& peek_data = *(peek_data_ptr);
  assert(top.size() == peek_data.size());
  for (unsigned int i = 0; i < top.size(); ++i)
  {
    vector<int> shape(peek_data[i]->shape());
    shape[0] = batch_size;
    top[i]->Reshape(shape);
  }
  
  DLOG(INFO) << "Initializing prefetch";
  StartInternalThread();
  DLOG(INFO) << "Prefetch initialized.";
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
  for (unsigned int i = 0; i < output.size(); ++i)
    CHECK(output[i]->count());

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

  /// Fill output
  for (unsigned int i = 0; i < batch_size; ++i)
  {
    timer.Start();
    /// Fetch one data sample from internal reader
    Container* data_ptr = reader_.full().pop("Waiting for data");
    const Container& data = (*data_ptr);
    read_time += timer.MicroSeconds();
    timer.Start();
    /// Copy data from new sample into output
    for (unsigned int i = 0; i < output.size(); ++i)
    {
      Blob<Dtype>* target_ptr = output[i];
      Blob<Dtype>* const source_ptr = data[i];
      const int offset = target_ptr->offset(i, 0, 0, 0);
      caffe_copy(source_ptr->count(),
                 source_ptr->cpu_data()+offset,
                 target_ptr->mutable_cpu_data()+offset);
    }
    trans_time += timer.MicroSeconds();
    
    /// Recycle spent data container for data reading
    reader_.free().push(data_ptr);
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
  (void)bottom;
  
  Container* container_ptr = prefetch_full_.pop("Data layer prefetch queue empty");
  const Container& container = (*container_ptr);

  /// Reshape tops and copy data
  for (unsigned int i = 0; i < top.size(); ++i) {
    top[i]->ReshapeLike(*container[i]);
    caffe_copy(container[i]->count(), 
               container[i]->cpu_data(),  
               top[i]->mutable_cpu_data());
    
  }
  
  /// Recycle spent data container for prefetching
  prefetch_free_.push(container_ptr);
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
