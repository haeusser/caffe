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
  
  LayerParameter param;
  DataParameter* data_param = param.mutable_data_param();
  //data_param->set_source("/scratch/global/hackathon/data");
  //data_param->set_clip_list("/scratch/global/hackathon/data/test_clip_list.txt");
  
  data_param->set_source("/misc/lmbraid17/sceneflownet/common/data4_bin-db");
  data_param->set_clip_list("/misc/lmbraid17/sceneflownet/common/data4_bin-db/cliplist_test01.txt");
  
  DataSample *sample1 = data_param->add_sample();
   DataEntry *ent1_1 = sample1->add_entry();
   ent1_1->set_name("imageL");
   ent1_1->set_offset(0);
   
   DataEntry *ent1_2 = sample1->add_entry();
   ent1_2->set_name("forwardFlowL");
   ent1_2->set_offset(0);
   
   DataEntry *ent1_3 = sample1->add_entry();
   ent1_3->set_name("dispL");
   ent1_3->set_offset(1);

   DataEntry *ent1_4 = sample1->add_entry();
   ent1_4->set_name("dispR");
   ent1_4->set_offset(-2);
  
  
  string param_str;
  param.SerializeToString(&param_str);
  
  Py_Initialize();
  try {
    bp::object module = bp::import("binarydb");
    bp::object dbclass = module.attr("BinaryDB")(param_str, 4);
    
    bp::list infos = (bp::list)dbclass.attr("getInfos")();
    
    /*int n = bp::len((infos));
    
    std::cout << "Len: " << n << std::endl;
    for(unsigned int i=0; i<n; i++){
      string str = boost::python::extract<string>((infos)[i]);
      std::cout << "String: " << str << std::endl;
    }*/
    
    //shared_ptr<Layer<Dtype> > = bp::extract<shared_ptr<PythonLayer<Dtype> > >(layer)();
    
  } catch (bp::error_already_set) {
    PyErr_Print();
    throw;
  }
  
  return 0;
}
