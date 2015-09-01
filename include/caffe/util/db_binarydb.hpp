#ifndef CAFFE_UTIL_DB_LMDB_HPP
#define CAFFE_UTIL_DB_LMDB_HPP

#include <string>

#include "caffe/util/db.hpp"

namespace caffe { namespace db {

inline void MDB_CHECK(int mdb_status) {
  CHECK_EQ(mdb_status, MDB_SUCCESS) << mdb_strerror(mdb_status);
}

class BinaryDBCursor {
 public:
  explicit BinaryDBCursor(BinaryDB *m_db)
    : m_db_(m_db), valid_(false) {
    SeekToFirst();
  }
  virtual ~BinaryDBCursor() {
    // TODO
  }
  virtual void SeekToFirst() { /*TODO*/ }
  virtual void Next() { /*TODO*/ }
  virtual string value() {
    return string(static_cast<const char*>(mdb_value_.mv_data),
        mdb_value_.mv_size);
  }
  virtual bool valid() { return valid_; }

 private:
  void Seek(MDB_cursor_op op) {
    int mdb_status = mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, op);
    if (mdb_status == MDB_NOTFOUND) {
      valid_ = false;
    } else {
      MDB_CHECK(mdb_status);
      valid_ = true;
    }
  }

  BinaryDB* m_db_;
  bool valid_;
};

class BinaryDB : public DB {
 public:
  BinaryDB() {
    
  }
  virtual ~BinaryDB() { Close(); }
  virtual void Open(const string& source, Mode mode);
  virtual void Close() {
    if (mdb_env_ != NULL) {
      mdb_dbi_close(mdb_env_, mdb_dbi_);
      mdb_env_close(mdb_env_);
      mdb_env_ = NULL;
    }
  }
  virtual BinaryDBCursor* NewCursor();

 private:
};

}  // namespace db
}  // namespace caffe

#endif  // CAFFE_UTIL_DB_LMDB_HPP
