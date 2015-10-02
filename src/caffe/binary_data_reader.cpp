#include <boost/thread.hpp>
#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/binary_data_reader.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/util/binarydb.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

using boost::weak_ptr;

template<> map<const string, weak_ptr<BinaryDataReader<float>::Body> > BinaryDataReader<float>::bodies_ = map<const string, weak_ptr<BinaryDataReader<float>::Body> >();
template<> map<const string, weak_ptr<BinaryDataReader<double>::Body> > BinaryDataReader<double>::bodies_ = map<const string, weak_ptr<BinaryDataReader<double>::Body> >();

static boost::mutex bodies_mutex_;

template <typename Dtype>
BinaryDataReader<Dtype>::BinaryDataReader(const LayerParameter& param)
  : queue_pair_(new BinaryQueuePair(  //
        param.data_param().prefetch() * param.data_param().batch_size(), param.top_size())),
  num_blobs_(param.top_size())
        {
  
  int queue_size = param.data_param().prefetch() * param.data_param().batch_size() * param.top_size();
  if(queue_size == 0) {
    LOG(FATAL) << "BinaryDataReader: One of prefetch, batch_size, top_size is 0!";
  }
  // Get or create a body
  boost::mutex::scoped_lock lock(bodies_mutex_);
  string key = source_key(param);
  weak_ptr<Body>& weak = bodies_[key];
  body_ = weak.lock();
  if (!body_) {
    body_.reset(new Body(param));
    bodies_[key] = weak_ptr<Body>(body_);
  }
  body_->new_queue_pairs_.push(queue_pair_);
}

template <typename Dtype>
BinaryDataReader<Dtype>::~BinaryDataReader() {
  string key = source_key(body_->param_);
  body_.reset();
  boost::mutex::scoped_lock lock(bodies_mutex_);
  if (bodies_[key].expired()) {
    bodies_.erase(key);
  }
}

template <typename Dtype>
BinaryDataReader<Dtype>::BinaryQueuePair::BinaryQueuePair(int size, int num_blobs) {
  // Initialize the free queue with requested number of blob vectors
  for (int i = 0; i < size; ++i) {
    vector<Blob<Dtype>*>* vec = new vector<Blob<Dtype>*>(num_blobs);
    for(int j = 0; j < num_blobs; j++) (*vec)[j] = new Blob<Dtype>(); // empty
    free_.push(vec);
  }
}

template <typename Dtype>
BinaryDataReader<Dtype>::BinaryQueuePair::~BinaryQueuePair() {
  vector<Blob<Dtype>*>* vec;
  while (free_.try_pop(&vec)) {
    for(int j = 0; j < vec->size(); j++) {
      if((*vec)[j]) delete (*vec)[j];
    }
    delete vec;
  }
  while (full_.try_pop(&vec)) {
    for(int j = 0; j < vec->size(); j++) {
      if((*vec)[j]) delete (*vec)[j];
    }
    delete vec;
  }
}

template <typename Dtype>
BinaryDataReader<Dtype>::Body::Body(const LayerParameter& param)
    : param_(param),
      new_queue_pairs_() {
  db_ = new db::BinaryDB<Dtype>;
  db_->Open(param.data_param().source(), param);

  if(db_->get_num_samples() < 1) {
    LOG(FATAL) << "No samples in DB";
  }

  StartInternalThread();
}

template <typename Dtype>
BinaryDataReader<Dtype>::Body::~Body() {
  StopInternalThread();
  db_->Close();
  delete db_; db_ = 0;
}

template <typename Dtype>
void BinaryDataReader<Dtype>::Body::InternalThreadEntry() {
  CHECK_EQ(param_.data_param().backend(), DataParameter_DB_BINARYDB);

  int index = 0;
  vector<shared_ptr<BinaryQueuePair> > qps;
  
  try {
    int solver_count = param_.phase() == TRAIN ? Caffe::solver_count() : 1;

    // To ensure deterministic runs, only start running once all solvers
    // are ready. But solvers need to peek on one item during initialization,
    // so read one item, then wait for the next solver.
    for (int i = 0; i < solver_count; ++i) {
      shared_ptr<BinaryQueuePair> qp(new_queue_pairs_.pop());
      read_one(index, qp.get());
      qps.push_back(qp);
    }
    // Main loop
    while (!must_stop()) {
      for (int i = 0; i < solver_count; ++i) {
        read_one(index, qps[i].get());
      }
      // Check no additional readers have been created. This can happen if
      // more than one net is trained at a time per process, whether single
      // or multi solver. It might also happen if two data layers have same
      // name and same source.
      CHECK_EQ(new_queue_pairs_.size(), 0);
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
}

template <typename Dtype>
void BinaryDataReader<Dtype>::Body::read_one(int &index, 
                                             BinaryQueuePair* qp) 
{
  while(true){
    
  std::queue<std::vector<Blob<Dtype>*>*> in_progress;
  
  const int max_parallel = param_.data_param().batch_size();

  Timer t;
  
  for (unsigned int i = 0; i < max_parallel; ++i) {
    vector<Blob<Dtype>*>* sample = qp->free_.pop();
    
    if (i == 0)
      t.Start();
    
    int compressed_size;
    db_->get_sample(index, sample, &compressed_size, i==max_parallel-1);
    
    if (i == max_parallel-1) {
      t.Stop();
      TimingMonitor::addMeasure("data_single_read", t.MilliSeconds());
      
      TimingMonitor::addMeasure("data_rate", compressed_size * 1000.0 / (t.MilliSeconds() * 1024.0 * 1024.0));
    }

    in_progress.push(sample);
    LOG(INFO) << "PF free/busy/full: " << qp->free_.size() << "/"
              << in_progress.size() << "/" << qp->full_.size();

    index++;
    if(index >= db_->get_num_samples()) {
      index = 0;
      DLOG(INFO) << "Restarting data prefetching from start.";
    }
  }
  
  while (in_progress.size() > 0) {
    qp->full_.push(in_progress.front());
    in_progress.pop();
  }
  }
}

INSTANTIATE_CLASS(BinaryDataReader);

}  // namespace caffe
