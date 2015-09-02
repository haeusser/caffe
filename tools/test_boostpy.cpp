// This is a script to upgrade "V0" network prototxts to the new format.
// Usage:
//    upgrade_net_proto_binary v0_net_proto_file_in net_proto_file_out

#include <cstring>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>

#include <boost/python.hpp>

#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

namespace bp = boost::python;

using std::ofstream;

using namespace caffe;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  
  Py_Initialize();
  try {
    bp::object module = bp::import("boostpython_test");
    bp::object layer = module.attr("Test1")();
    
    bp::list ret = (bp::list)layer.attr("method")(5);
    
    int n = bp::len((ret));
    
    std::cout << "Len: " << n << std::endl;
    for(unsigned int i=0; i<n; i++){
      string str = boost::python::extract<string>((ret)[i]);
      std::cout << "String: " << str << std::endl;
    }
    
    //shared_ptr<Layer<Dtype> > = bp::extract<shared_ptr<PythonLayer<Dtype> > >(layer)();
    
  } catch (bp::error_already_set) {
    PyErr_Print();
    throw;
  }
  
  return 0;
}
