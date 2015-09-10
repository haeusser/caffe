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
    
    // returns (all_samples, entry_dimensions, bin_filenames)
    bp::tuple infos = (bp::tuple)dbclass.attr("getInfos")();
    
    if(bp::len(infos) != 3) LOG(FATAL) << "Python did not return 3-tuple";
    
    bp::list all_samples = (bp::list)infos[0];
    bp::list dimensions = (bp::list)infos[1];
    bp::list bin_filenames = (bp::list)infos[2];
    
    // Store dimensions:
    if(bp::len(dimensions) != top_num_) LOG(FATAL) << "Number of entry dimensions passed from python not equal to top blob count";
    
    entry_dimensions_.resize(top_num_);
    for(int entry = 0; entry < top_num_; entry++) {
      entry_dimensions_[entry].resize(4);
      entry_dimensions_[entry][0] = 1;
      entry_dimensions_[entry][1] = boost::python::extract<int>(dimensions[entry][2]);
      entry_dimensions_[entry][2] = boost::python::extract<int>(dimensions[entry][1]);
      entry_dimensions_[entry][3] = boost::python::extract<int>(dimensions[entry][0]);
    }
    
    // Store bin filenames:
    binfiles_.resize(bp::len(bin_filenames));
    for(int i=0; i<bp::len(bin_filenames); i++) {
      binfiles_[i] = boost::python::extract<string>(bin_filenames[i]);
    }
    
    // Store samples
    // [[(0, 3110400, 1), (1, 4147200, 3), (2, 3110400, 3)], [(),(),()], ... ]
    num_samples_ = bp::len(all_samples);
    samples_.resize(num_samples_);
    for(int sample=0; sample<num_samples_; sample++) {
      samples_[sample].resize(top_num_);
      for(int entry=0; entry<top_num_; entry++) {
        samples_[sample][entry].binfile_idx = boost::python::extract<int>(all_samples[sample][entry][0]);
        samples_[sample][entry].byte_offset = boost::python::extract<long int>(all_samples[sample][entry][1]);
        int encoding_int = boost::python::extract<int>(all_samples[sample][entry][2]);
        samples_[sample][entry].data_encoding = (BinaryDB_DataEncoding)encoding_int;
      }
    }
    
  } catch (bp::error_already_set) {
    PyErr_Print();
    throw;
  }
  
  // open binfiles
  binstreams_.resize(binfiles_.size());
  for (int i = 0; i < binfiles_.size(); ++i) {
    binstreams_.at(i).reset(new std::ifstream(binfiles_.at(i).c_str(), std::ios::in | std::ios::binary));
    if(!binstreams_.at(i)->is_open() || !binstreams_.at(i)->good()) {
      LOG(FATAL) << "Could not open bin file " << binfiles_.at(i);
    }
  }


  // DEBUG
//   int index = 792;
//   int t = 0;
//   LOG(INFO) << "Before shuffle: sample " << index << " blob " << t << " comes from bin file " << binfiles_.at(samples_.at(index).at(t).binfile_idx) << " offset " << samples_.at(index).at(t).byte_offset;
  
  // permute the samples
  if (param.data_param().rand_permute()) {  
    int seed = param.data_param().rand_permute_seed();
    if(seed > 0) std::srand (unsigned(seed));
    std::random_shuffle ( samples_.begin(), samples_.end() );  
  }
  
  // DEBUG  
//   LOG(FATAL) << "After shuffle: sample " << index << " blob " << t << " comes from bin file " << binfiles_.at(samples_.at(index).at(t).binfile_idx) << " offset " << samples_.at(index).at(t).byte_offset;
  
  LOG(INFO) << "Opened BinaryDB " << source;
}


template <typename Dtype>
void BinaryDB<Dtype>::Close() {
  for (int i = 0; i < binfiles_.size(); ++i)
    binstreams_.at(i)->close();
}

template <typename Dtype>
void BinaryDB<Dtype>::get_sample(int index, vector<Blob<Dtype>*>* dst) {
  // loop through top blobs
  for (int t=0; t<top_num_; ++t) {
    Entry entry = samples_.at(index).at(t);
    std::ifstream* curr_binstream = binstreams_.at(entry.binfile_idx).get();
    curr_binstream->seekg(entry.byte_offset, ios::beg);
    
    // check if the stream is ok, re-open if needed
    if (!curr_binstream->is_open() || !curr_binstream->good()) {
      LOG(INFO) << "Smth wrong eih the stream of file " << binfiles_.at(entry.binfile_idx);
      
      LOG(INFO) << " is_open()=" << curr_binstream->is_open();
      LOG(INFO) << " good()=" << curr_binstream->good();
      LOG(INFO) << " eof()=" << curr_binstream->eof();
      LOG(INFO) << " fail()=" << curr_binstream->fail();
      LOG(INFO) << " bad()=" << curr_binstream->bad();
      
      LOG(INFO) << "Attempting to re-open";
      binstreams_.at(entry.binfile_idx).get()->close();
      binstreams_.at(entry.binfile_idx).get()->open(binfiles_.at(entry.binfile_idx).c_str(), std::ios::in | std::ios::binary);
      
      if(!curr_binstream->is_open() || !curr_binstream->good()) 
        LOG(FATAL) << "Could not re-open bin file " << binfiles_.at(entry.binfile_idx);
      else {
        curr_binstream = binstreams_.at(entry.binfile_idx).get();
        curr_binstream->seekg(entry.byte_offset, ios::beg);
      }
        
    }
    
    // check the flag of the entry (4-byte thing)
    unsigned int flag;
    curr_binstream->read(reinterpret_cast<char *>(&flag), 4);
    if (flag == 0)
      LOG(FATAL) << "Flag of blob " << t << " of sample " << index << " is 0 "
         << " (File " << binfiles_.at(entry.binfile_idx) << " offset " << entry.byte_offset << ")";
    else if (flag != 1)
      LOG(FATAL) << "Flag of blob " << t << " of sample " << index << " has invalid value " << flag
         << " (File " << binfiles_.at(entry.binfile_idx) << " offset " << entry.byte_offset << ")";
    
    // reshape blob
    dst->at(t)->Reshape(entry_dimensions_[t]);
    
    // actually read the data
    read_binstream(curr_binstream, entry.data_encoding, dst->at(t)->count(), dst->at(t)->mutable_cpu_data());
  }
}

// given a ifstream (with file pointer set correctly), reads N values from the stream to out with the given data_encoding
template <typename Dtype>
void BinaryDB<Dtype>::read_binstream(std::ifstream* binstream, BinaryDB_DataEncoding data_encoding, long int N, Dtype* out) {
  switch(data_encoding)
  {
    case BinaryDB_DataEncoding_UINT8:
      for(int i=0; i<N; i++) {
        unsigned char c;
        binstream->read(reinterpret_cast<char*>(&c), 1);
        *(out++)=static_cast<Dtype>(c);
      }
      break;
    case BinaryDB_DataEncoding_FIXED16DIV32:
      for(int i=0; i<N; i++) {
        short v;
        binstream->read(reinterpret_cast<char*>(&v), 2);

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
