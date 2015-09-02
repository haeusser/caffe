// Check the reading of binary files by binarydb functions

#include <cstring>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>

#include <boost/python.hpp>

#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/output.hpp"
#include "caffe/util/upgrade_proto.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

template <typename Dtype>
void read_binstream(std::ifstream* binstream, BinaryDB_DataEncoding data_encoding, long int N, Dtype* out);

int main(int argc, char** argv) {
  
  int nimg;
  if (argc == 1)
    nimg = 0;
  else
    nimg = atoi(argv[1]);
  
  printf("nimg=%d\n", nimg);
  
  // check reading an image
  
  printf("\n == Reading the image == \n");
  
  vector<int> shape;
  shape.push_back(1);
  shape.push_back(3);
  shape.push_back(540);
  shape.push_back(960);  
  
  std::ifstream binstream("/misc/lmbraid17/sceneflownet/common/data4_bin-db/lowres/final/treeflight/image.bin", 
                          std::ios::in | std::ios::binary);
  
  Blob<float> data;
  data.Reshape(shape); 
  
  binstream.seekg((data.count()+4)*nimg*2, ios::beg);
  
  // check the flag of the entry (4-byte thing)
  unsigned int flag;
  binstream.read(reinterpret_cast<char *>(&flag), 4);
  printf("Flag: %d\n", flag);
 
  read_binstream(&binstream, BinaryDB_DataEncoding_UINT8, data.count(), data.mutable_cpu_data());
  
  writePPM("/misc/lmbraid17/sceneflownet/dosovits/test.ppm", data.cpu_data(), shape.at(3), shape.at(2), true, 1.);
  
  printf("\n");
  
  for(int i=0; i<100; i++)
    printf("%f ", (data.cpu_data())[i*(data.count()/100)]);
  
  // check reading a flow
  printf("\n\n == Reading the flow == \n");
  
    
  binstream.close();
  binstream.open("/misc/lmbraid17/sceneflownet/common/data4_bin-db/lowres/final/treeflight/flow.bin", 
                 std::ios::in | std::ios::binary);
  
  
  shape.at(0) = 1;
  shape.at(1) = 2;
  shape.at(2) = 540;
  shape.at(3) = 960; 
  data.Reshape(shape); 
  
  binstream.seekg((data.count()*2+4)*nimg, ios::beg);
  
  // check the flag of the entry (4-byte thing)
  binstream.read(reinterpret_cast<char *>(&flag), 4);
  printf("Flag: %d\n", flag);
 
  read_binstream(&binstream, BinaryDB_DataEncoding_FIXED16DIV32, data.count(), data.mutable_cpu_data());
  
  writeFloFile("/misc/lmbraid17/sceneflownet/dosovits/test.flo", data.cpu_data(), shape.at(3), shape.at(2));
  
  printf("\n");
  
  for(int i=0; i<100; i++)
    printf("%f ", (data.cpu_data())[i*(data.count()/100)]);
    
  return 0;
}

template <typename Dtype>
void read_binstream(std::ifstream* binstream, BinaryDB_DataEncoding data_encoding, long int N, Dtype* out) {
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
        if (!(i%(N/100)))
          printf("%d ", v);

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
