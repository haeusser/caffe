// Copyright 2014 BVLC and contributors.

#include <stdint.h>
#include <leveldb/db.h>
#include <pthread.h>

#include <fstream>  // NOLINT(readability/streams)
#include <vector>

#include <boost/thread.hpp>

/// Caffe and local files
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/binary_data_reader.hpp"
#include "caffe/net.hpp"


namespace caffe {

const int kMaxKeyLength = 256;

#define Container vector<Blob<Dtype>*>


// template <typename Dtype>
// void* BinaryDataLayerPrefetch(void* layer_pointer) {
//   CHECK(layer_pointer);
//   BinaryDataLayer<Dtype>* layer = static_cast<BinaryDataLayer<Dtype>*>(layer_pointer);
//   CHECK(layer);
//   Datum datum;
//   const Dtype scale = layer->layer_param_.data_param().scale();
//   const int batch_size = layer->layer_param_.data_param().batch_size();
//   const int crop_size = layer->layer_param_.data_param().crop_size();
//   const bool mirror = layer->layer_param_.data_param().mirror();
// 
//   if (mirror && crop_size == 0) {
//     LOG(FATAL) << "Current implementation requires mirror and crop_size to be "
//         << "set at the same time.";
//   }
//   // datum scales
//   const int channels = layer->datum_channels_;
//   const int height = layer->datum_height_;
//   const int width = layer->datum_width_;
//   //const int size = layer->datum_size_;
//   
//   const int heightwidth = height*width;
//   
//   const Dtype* mean = layer->data_mean_.cpu_data();
//   for (int item_id = 0; item_id < batch_size; ++item_id) {
//     // go to the next iter
//     switch (layer->layer_param_.data_param().backend()) {
//     case DataParameter_DB_LEVELDB:
//       LOG(FATAL) << "LevelDB not supported by BinaryData";
//       break;
//     case DataParameter_DB_LMDB: {
//       if(layer->datum_index_ >= layer->range_size_) layer->datum_index_ = 0; //wrap around
//       
//       int dbIndex = layer->permutation_vector_.at(layer->datum_index_); // optionally shifted and permuted position (range and rand_perm)
//       
//       //LOG(INFO) << "Fetching: datum " << layer->datum_index_ << "/" << layer->range_size_ << ". I.e. permuted&ranged: " << dbIndex << ". ";
//       
//       char key_cstr[kMaxKeyLength];
//       snprintf(key_cstr, kMaxKeyLength, "%08d", dbIndex);
//       layer->mdb_key_.mv_data = (void*)key_cstr;
//       layer->mdb_key_.mv_size = strlen((char*)layer->mdb_key_.mv_data);
//       if(mdb_cursor_get(layer->mdb_cursor_, &layer->mdb_key_, &layer->mdb_value_, MDB_SET_RANGE) != MDB_SUCCESS) {
//         LOG(FATAL) << "Internal data fetch error: Tried to fetch element " << layer->datum_index_ << " of " << layer->range_size_ << " which is in DB: " << dbIndex;
//       }
//       
//       layer->datum_index_++;
//       
//     } break;
//     default:
//       LOG(FATAL) << "Unknown database backend";
//     }
// 
//     // get a blob
//     switch (layer->layer_param_.data_param().backend()) {
//     case DataParameter_DB_LEVELDB:
//       LOG(FATAL) << "LevelDB not supported by BinaryData";
//       break;
//     case DataParameter_DB_LMDB:
//       CHECK_EQ(mdb_cursor_get(layer->mdb_cursor_, &layer->mdb_key_,
//               &layer->mdb_value_, MDB_GET_CURRENT), MDB_SUCCESS);
//       datum.ParseFromArray(layer->mdb_value_.mv_data,
//           layer->mdb_value_.mv_size);
//       break;
//     default:
//       LOG(FATAL) << "Unknown database backend";
//     }
// 
//     // Iterate over slices:
//     int src_channel_start = -1; //inclusive
//     int src_channel_end = 0; //non-inclusive (end will become start in next slice)
//     
//     Dtype* decoded_data;
//     DecodeData(decoded_data,datum,layer->slice_point_,layer->channel_encoding_); // Can deal with all types of data
// 
//     for(int slice = 0; slice <= layer->slice_point_.size(); slice++) {
//         src_channel_start = src_channel_end;
// 
//         if(slice == layer->slice_point_.size()) { // last slice
//             src_channel_end = channels;
//         } else {
//             src_channel_end = layer->slice_point_[slice];
//         }
//         
//         CHECK(layer->prefetch_data_blobs_[slice]);
//         Dtype* top_data = layer->prefetch_data_blobs_[slice]->mutable_cpu_data();
// 
//         int slice_channel_count = src_channel_end - src_channel_start;
// 
//         if (crop_size) {
//             int h_off, w_off;
//             // We only do random crop when we do training.
//             /*if (layer->phase_ == Caffe::TRAIN) {
//           h_off = layer->PrefetchRand() % (height - crop_size);
//           w_off = layer->PrefetchRand() % (width - crop_size);
//           } else {*/
//             // Always do fixed crop (also when we do training)
//             h_off = (height - crop_size) / 2;
//             w_off = (width - crop_size) / 2;
//             //}
//             if (mirror && layer->PrefetchRand() % 2) {
//                 // Copy mirrored version
//                 for (int c = 0; c < slice_channel_count; ++c) {
//                     for (int h = 0; h < crop_size; ++h) {
//                         for (int w = 0; w < crop_size; ++w) {
//                             int srcc = src_channel_start + c;
// 
//                             int top_index = ((item_id * slice_channel_count + c) * crop_size + h)
//                                     * crop_size + (crop_size - 1 - w);
//                             int data_index = (srcc * height + h + h_off) * width + w + w_off;
//                             Dtype datum_element = decoded_data[data_index];
//                             top_data[top_index] = (datum_element - mean[data_index]) * scale;
//                         }
//                     }
//                 }
//             } else {
//                 // Normal copy
//                 for (int c = 0; c < slice_channel_count; ++c) {
//                     for (int h = 0; h < crop_size; ++h) {
//                         for (int w = 0; w < crop_size; ++w) {
//                             int srcc = src_channel_start + c;
// 
//                             int top_index = ((item_id * slice_channel_count + c) * crop_size + h)
//                                     * crop_size + w;
//                             int data_index = (srcc * height + h + h_off) * width + w + w_off;
//                             Dtype datum_element = decoded_data[data_index];
//                             top_data[top_index] = (datum_element - mean[data_index]) * scale;
//                         }
//                     }
//                 }
//             }
//         } else {
//             // we will prefer to use data() first if existant, and then try float_data()
//             for (int c = 0; c < slice_channel_count; ++c) {
//                 for (int hw = 0; hw < heightwidth; ++hw) {
//                     int srcc = src_channel_start + c;
// 
//                     int top_index = (item_id * slice_channel_count + c) * heightwidth + hw;
//                     int data_index = srcc * heightwidth + hw;
// 
//                     Dtype datum_element = decoded_data[data_index];
//                     top_data[top_index] = (datum_element - mean[data_index]) * scale;
//                 }
//             }
//         }
// 
//         //DEBUG:
//         /*std::stringstream sstm;
//       sstm << "debugout." << layer->layer_param_.name() << "." << item_id << "." << layer->mdb_cursor_ << ".txt";
//       std::ofstream outfile(sstm.str().c_str());
//       for (int j = 0; j < size; ++j) {
//             outfile << top_data[item_id * size + j] << std::endl;
//       }*/
// 
//     } //end-for slice iterator
//     free(decoded_data);
// 
//   } //end-for batch item
// 
//   return static_cast<void*>(NULL);
// }


/**
 * @brief Constructor
 * 
 * @note This method only calls the base class constructor
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
  //const DataParameter& data_param = this->layer_param_.data_param();
  
  const bool verbose = this->layer_param_.data_param().verbose();
  if (verbose && this->layer_param_.data_param().block_size())
    LOG(INFO) << "Block size: " << this->layer_param_.data_param().block_size();

  /// Read a data point, and use it to initialize the top blob.
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
 * @brief Reshape
 */
template <typename Dtype>
void BinaryDataLayer<Dtype>::Reshape(const Container& bottom,
                                     const Container& top) 
{
  /// TODO what does this do?
}


/**
 * @brief Forward_cpu
 * TODO
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
