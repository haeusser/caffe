#include "caffe/util/binarydb.hpp"

#include <string>

#include <boost/python.hpp>

namespace bp = boost::python;

namespace caffe { namespace db {

template <typename Dtype>
void BinaryDB<Dtype>::Open(const string& source, const LayerParameter& param) {
  top_num_ = param.top_size();
  sample_variants_num_ = param.data_param().sample_size();
  
  LOG(INFO) << "Opening BinaryDB using boost::python";
  
  string param_str;
  param.SerializeToString(&param_str);
  
  Py_Initialize();
  try {
    bp::object module = bp::import("binarydb");
    bp::object dbclass = module.attr("BinaryDB")(param_str, top_num_);
    
    bp::list infos = (bp::list)dbclass.attr("getInfos")();
    
//     int n = bp::len((ret));
//     
//     std::cout << "Len: " << n << std::endl;
//     for(unsigned int i=0; i<n; i++){
//       string str = boost::python::extract<string>((ret)[i]);
//       std::cout << "String: " << str << std::endl;
//     }
  } catch (bp::error_already_set) {
    PyErr_Print();
    throw;
  }
  
  // open binfiles
  binstreams_.resize(binfiles_.size());
  for (int i = 0; i < binfiles_.size(); ++i)
    binstreams_.at(i) = new std::ifstream(binfiles_.at(i).c_str(), std::ios::in | std::ios::binary);

  // permute the samples  
  std::random_shuffle ( samples_.begin(), samples_.end() );  
  
  LOG(INFO) << "Opened BinaryDB " << source;
}


template <typename Dtype>
void BinaryDB<Dtype>::Close() {
  //TODO
}

template <typename Dtype>
void BinaryDB<Dtype>::get_sample(int index, vector<Blob<Dtype>*>* dst) {
  // loop through top blobs
  for (int t=0; t<top_num_; ++t) {
    Entry entry = samples_.at(index).at(t);
    std::ifstream* curr_binstream = binstreams_.at(entry.binfile_idx);
    curr_binstream->seekg(entry.byte_offset);
    
    // check the flag of the entry (4-byte thing)
    int flag;
    curr_binstream->read(reinterpret_cast<char *>(&flag), 4);
    if (flag == 1)
      LOG(FATAL) << "Blob " << t << "of sample " << index << " not accessible";
    else if (flag != 0)
      LOG(FATAL) << "Flag of blob " << t << "of sample " << index << " has invalid value " << flag;    
    
    // actually read the data
    read_values(curr_binstream, entry.data_encoding, dst->at(t)->count(), dst->at(t)->mutable_cpu_data());
  }
}

// given a ifstream (with file pointer set correctly), reads N values from the stream to out with the given data_encoding
template <typename Dtype>
void BinaryDB<Dtype>::read_values(std::ifstream* binstream, BinaryDB_DataEncoding data_encoding, long int N, Dtype* out) {
  switch(data_encoding)
  {
    case BinaryDB_DataEncoding_UINT8:
      for(int i=0; i<N; i++) {
        char c;
        binstream->read(&c, 1);
        *(out++)=static_cast<Dtype>(c);
      }
      break;
    case BinaryDB_DataEncoding_FIXED16DIV32:
      for(int i=0; i<N; i++) {
        short v;
        binstream->read(reinterpret_cast<char *>(&v), 2);

        Dtype value;
        if(v==std::numeric_limits<short>::max()) 
          value = std::numeric_limits<Dtype>::signaling_NaN();
        else
          value = ((Dtype)v)/32.0;

        *(out++)=value;
      }
      break;
    default:
        LOG(FATAL) << "Unknown data encoding " << data_encoding;
        break;
  }
}


INSTANTIATE_CLASS(BinaryDB);

}  // namespace db
}  // namespace caffe
