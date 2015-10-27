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
  
  Blob<float>* test_blob = new Blob<float>(2,1,5,5); 
  
  int H = test_blob->shape()[2];
  int W = test_blob->shape()[3];
  
  FillerParameter filler_param;
  
  LOG(INFO) << "==== Test 1 ====";
  filler_param.set_std(1.);
  GaussianFiller<float> gauss_filler(filler_param);
  gauss_filler.Fill(test_blob);  
  test_blob->print("Random Gaussian blob");
  
  Blob<float>* spectral_blob = man->SpatialToSpectral(Caffe::GPU, test_blob, SpectralComponentsManager<float>::BLOB_DATA);
  spectral_blob->print("Converted to spectral");
  
  Blob<float>* spatial_blob = new Blob<float>(test_blob->shape());   
  man->SpectralToSpatial(Caffe::GPU, spectral_blob, spatial_blob, SpectralComponentsManager<float>::BLOB_DATA);  
  spatial_blob->print("Converted back to spatial");
  
  LOG(INFO) << "==== Test 2 ====";
  filler_param.set_value(0.);
  ConstantFiller<float> const_filler(filler_param);
  const_filler.Fill(test_blob);  
  float* test_data = test_blob->mutable_cpu_data();
  for (int y=0; y<H; y++)
    for (int x=0; x<W; x++)
      if (x == W/2)
        test_data[W*y + x] = 1.; 
  
  test_blob->print("Blob with a line");
  spectral_blob = man->SpatialToSpectral(Caffe::GPU, test_blob, SpectralComponentsManager<float>::BLOB_DATA);
  spectral_blob->print("Converted to spectral");
  
  man->SpectralToSpatial(Caffe::GPU, spectral_blob, spatial_blob, SpectralComponentsManager<float>::BLOB_DATA);  
  spatial_blob->print("Converted back to spatial");
  
  LOG(INFO) << "==== Test 3 ====";
  test_blob = man->getOrMakeBank(W,H);
  test_blob->print("The filter bank");
  
  spectral_blob = man->SpatialToSpectral(Caffe::GPU, test_blob, SpectralComponentsManager<float>::BLOB_DATA);
  spectral_blob->print("Converted to spectral");
  
  man->SpectralToSpatial(Caffe::GPU, spectral_blob, spatial_blob, SpectralComponentsManager<float>::BLOB_DATA);  
  spatial_blob->print("Converted back to spatial");
  
  delete man;
  
  
  
  return 0;
}
