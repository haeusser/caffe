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
  LOG(INFO) << "Opened BinaryDB " << source;
}


template <typename Dtype>
void BinaryDB<Dtype>::Close() {
  //TODO
}

template <typename Dtype>
int BinaryDB<Dtype>::get_num_samples() {
  return 0; // TODO
}

template <typename Dtype>
void BinaryDB<Dtype>::get_sample(int index, vector<Blob<Dtype>*>* dst) {
  //TODO
}


INSTANTIATE_CLASS(BinaryDB);

}  // namespace db
}  // namespace caffe
