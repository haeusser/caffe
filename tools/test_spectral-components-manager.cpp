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

using namespace caffe;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  
  SpectralComponentsManager<float> *man = new SpectralComponentsManager<float>();
  
  delete man;
  
  
  
  return 0;
}
