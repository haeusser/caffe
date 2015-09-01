#ifndef CAFFE_DATA_READER_HPP_
#define CAFFE_DATA_READER_HPP_

#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/db.hpp"

namespace caffe {

/**
 * @brief Reads data from a source to queues available to data layers.
 * A single reading thread is created per source, even if multiple solvers
 * are running in parallel, e.g. for multi-GPU training. This makes sure
 * databases are read sequentially, and that each solver accesses a different
 * subset of the database. Data is distributed to solvers in a round-robin
 * way to keep parallel training deterministic.
 */
class BinaryDataReader : public InternalThread {
 public:
  explicit BinaryDataReader(const LayerParameter& param);
  ~BinaryDataReader();

  inline BlockingQueue<vector<Blob<Dtype> >*>& free() const {
    return queue_pair_->free_;
  }
  inline BlockingQueue<vector<Blob<Dtype> >*>& full() const {
    return queue_pair_->full_;
  }
  const LayerParameter param_;
    
 protected:
  // Queue pairs are shared between a body and its readers
  class BinaryQueuePair {
   public:
    explicit BinaryQueuePair(int size);
    ~BinaryQueuePair();

    BlockingQueue<vector<Blob<Dtype> >*> free_;
    BlockingQueue<vector<Blob<Dtype> >*> full_;

  DISABLE_COPY_AND_ASSIGN(BinaryQueuePair);
  };

  // A single body is created per source
  class Body : public InternalThread {
   public:
    explicit Body(const LayerParameter& param);
    virtual ~Body();

   protected:
    void InternalThreadEntry();
    void read_one(int cursor, QueuePair* qp);

    const LayerParameter param_;
    BlockingQueue<shared_ptr<QueuePair> > new_queue_pairs_;

    friend class BinaryDataReader;

  DISABLE_COPY_AND_ASSIGN(Body);
  };

  // A source is uniquely identified by its layer name + path, in case
  // the same database is read from two different locations in the net.
  static inline string source_key(const LayerParameter& param) {
    return param.name() + ":" + param.data_param().source();
  }

  const shared_ptr<QueuePair> queue_pair_;
  shared_ptr<Body> body_;

  static map<const string, boost::weak_ptr<BinaryDataReader::Body> > bodies_;

DISABLE_COPY_AND_ASSIGN(BinaryDataReader);
};

}  // namespace caffe

#endif  // CAFFE_DATA_READER_HPP_