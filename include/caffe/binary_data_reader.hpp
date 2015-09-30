#ifndef CAFFE_BINARY_DATA_READER_HPP_
#define CAFFE_BINARY_DATA_READER_HPP_

#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/db.hpp"
#include "caffe/util/binarydb.hpp"

namespace caffe {

/**
 * @brief Reads data from a source to queues available to data layers.
 * A single reading thread is created per source, even if multiple solvers
 * are running in parallel, e.g. for multi-GPU training. This makes sure
 * databases are read sequentially, and that each solver accesses a different
 * subset of the database. Data is distributed to solvers in a round-robin
 * way to keep parallel training deterministic.
 */

template <typename Dtype>
class BinaryDataReader : public InternalThread {
 public:
  explicit BinaryDataReader(const LayerParameter& param);
  ~BinaryDataReader();

  inline BlockingQueue< vector<Blob<Dtype>*>*>& free() const {
    return queue_pair_->free_;
  }
  inline BlockingQueue< vector<Blob<Dtype>*>*>& full() const {
    return queue_pair_->full_;
  }
  const LayerParameter param_;

  int get_num_samples() { return body_.get()->get_num_samples(); }

 protected:
  // Queue pairs are shared between a body and its readers
  class BinaryQueuePair {
   public:
    explicit BinaryQueuePair(int size, int num_blobs);
    ~BinaryQueuePair();

    BlockingQueue< vector<Blob<Dtype>*>*> free_;
    BlockingQueue< vector<Blob<Dtype>*>*> full_;

  DISABLE_COPY_AND_ASSIGN(BinaryQueuePair);
  };

  // A single body is created per source
  class Body : public InternalThread {
   public:
    explicit Body(const LayerParameter& param);
    virtual ~Body();

    int get_num_samples() { return db_->get_num_samples(); }

   protected:
    void InternalThreadEntry();
    void read_one(int &index, BinaryQueuePair* qp);

    const LayerParameter param_;
    BlockingQueue<shared_ptr<BinaryQueuePair> > new_queue_pairs_;

    db::BinaryDB<Dtype>* db_;
    friend class BinaryDataReader;

  DISABLE_COPY_AND_ASSIGN(Body);
  };

  // A source is uniquely identified by its layer name + path, in case
  // the same database is read from two different locations in the net.
  static inline string source_key(const LayerParameter& param) {
    return param.name() + ":" + param.data_param().collection_list();
  }

  const shared_ptr<BinaryQueuePair> queue_pair_;
  shared_ptr<Body> body_;

  static map<const string, boost::weak_ptr<BinaryDataReader<Dtype>::Body> > bodies_;
  
  int num_blobs_;

DISABLE_COPY_AND_ASSIGN(BinaryDataReader);
};


}  // namespace caffe

#endif  // CAFFE_DATA_READER_HPP_
