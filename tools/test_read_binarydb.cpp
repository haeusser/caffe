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


typedef unsigned char uchar;

static int ncols = 0;
#define MAXCOLS 60
static int colorwheel[MAXCOLS][3];

static void setcols(int r, int g, int b, int k);
static void makecolorwheel(void);
static void sintelCartesianToRGB(float fx, float fy, float* pix);

template <typename Dtype>
void flow_to_color(int width, int height, const Dtype* flow, Dtype* color, Dtype scale);

template <typename Dtype>
void disp_to_color(int width, int height, const Dtype* disp, Dtype* out_img, Dtype scale);

using namespace caffe;  // NOLINT(build/namespaces)

template <typename Dtype>
void read_binstream(std::ifstream* binstream, BinaryDB_DataEncoding data_encoding, long int N, Dtype* out);

int main(int argc, char** argv) {
  
  long int nimg = 0;
  std::string data_path = "/misc/lmbraid17/sceneflownet/common/data/4_bin-db/lowres/final/treeflight";
  if (argc >= 2)
    nimg = atoi(argv[1]);
  if (argc == 3)
    data_path = argv[2];
  if (argc >= 4)
    LOG(FATAL) << "Usage: test_read_binarydb [image number] [data path]";
  
  std::cout << "\nData path: " << data_path << "\nImage number " << nimg << "\n";
  
  /// check reading images
  
  std::cout << "\n == Reading the image == \n";
  
  vector<int> shape;
  shape.push_back(1);
  shape.push_back(3);
  shape.push_back(540);
  shape.push_back(960);  
  
  long int offset;
  unsigned int flag;
  Blob<float> data, out_img;
  data.Reshape(shape); 
  out_img.Reshape(shape); 
  
  
  std::string path = data_path + "/image.bin";
  
  std::ifstream binstream(path.c_str(), 
                          std::ios::in | std::ios::binary);
  
  if (binstream.is_open()) {   
    
    offset = (long int)(data.count()+4)*nimg*2;
    std::cout << "\n Offset = " << offset << "\n";
    binstream.seekg(offset, ios::beg);
    
    // check the flag of the entry (4-byte thing)
    binstream.read(reinterpret_cast<char *>(&flag), 4);
    std::cout << "Flag L: " << flag << std::endl;
  
    read_binstream(&binstream, BinaryDB_DataEncoding_UINT8, data.count(), data.mutable_cpu_data());  
    writePPM("/misc/lmbraid17/sceneflownet/common/sandbox/test_imageL.ppm", data.cpu_data(), shape.at(3), shape.at(2), true, 1.);
    for(int i=0; i<10; i++)
      std::cout << (data.cpu_data())[i*(data.count()/10)] << " ";
    
    std::cout << std::endl;
    
    // check the flag of the entry (4-byte thing)
    binstream.read(reinterpret_cast<char *>(&flag), 4);
    std::cout << "Flag R: " << flag << std::endl;  
    read_binstream(&binstream, BinaryDB_DataEncoding_UINT8, data.count(), data.mutable_cpu_data());  
    writePPM("/misc/lmbraid17/sceneflownet/common/sandbox/test_imageR.ppm", data.cpu_data(), shape.at(3), shape.at(2), true, 1.);
    for(int i=0; i<10; i++)
      std::cout << (data.cpu_data())[i*(data.count()/10)] << " ";
    std::cout << std::endl;
  } else 
    LOG(WARNING) << "Failed to open " << path;
  
  /// check reading flows
  std::cout << "\n\n == Reading the flow == \n";  
  
  float flow_scale = 0.2;
    
  binstream.close();
  path = data_path + "/flow.bin";
  binstream.open(path.c_str(), 
                 std::ios::in | std::ios::binary);
  
  if (binstream.is_open()) {  
  
    shape.at(0) = 1;
    shape.at(1) = 2;
    shape.at(2) = 540;
    shape.at(3) = 960; 
    data.Reshape(shape); 
    
    offset = (long int)(data.count()*2+4)*nimg*4;
    std::cout << "\n Offset = " << offset << "\n";
    binstream.seekg(offset, ios::beg);
    
    // flow_forwardL
    binstream.read(reinterpret_cast<char *>(&flag), 4);
    std::cout << "Flag forwardL: " << flag << std::endl; 
    read_binstream(&binstream, BinaryDB_DataEncoding_FIXED16DIV32, data.count(), data.mutable_cpu_data());  
    flow_to_color<float>(shape.at(3), shape.at(2), data.cpu_data(), out_img.mutable_cpu_data(), flow_scale);  
    writePPM("/misc/lmbraid17/sceneflownet/common/sandbox/test_flow_forwardL.ppm", out_img.cpu_data(), shape.at(3), shape.at(2), true, 1.);  
    for(int i=0; i<10; i++)
      std::cout << (data.cpu_data())[i*(data.count()/10)] << " ";  
    std::cout << std::endl;
    
    // flow_forwardR
    binstream.read(reinterpret_cast<char *>(&flag), 4);
    std::cout << "Flag forwardR: " << flag << std::endl; 
    read_binstream(&binstream, BinaryDB_DataEncoding_FIXED16DIV32, data.count(), data.mutable_cpu_data());  
    flow_to_color<float>(shape.at(3), shape.at(2), data.cpu_data(), out_img.mutable_cpu_data(), flow_scale);  
    writePPM("/misc/lmbraid17/sceneflownet/common/sandbox/test_flow_forwardR.ppm", out_img.cpu_data(), shape.at(3), shape.at(2), true, 1.);  
    for(int i=0; i<10; i++)
      std::cout << (data.cpu_data())[i*(data.count()/10)] << " ";  
    std::cout << std::endl;
    
    // flow_backwardL
    binstream.read(reinterpret_cast<char *>(&flag), 4);
    std::cout << "Flag backwardL: " << flag << std::endl; 
    read_binstream(&binstream, BinaryDB_DataEncoding_FIXED16DIV32, data.count(), data.mutable_cpu_data());  
    flow_to_color<float>(shape.at(3), shape.at(2), data.cpu_data(), out_img.mutable_cpu_data(), flow_scale);  
    writePPM("/misc/lmbraid17/sceneflownet/common/sandbox/test_flow_backwardL.ppm", out_img.cpu_data(), shape.at(3), shape.at(2), true, 1.);  
    for(int i=0; i<10; i++)
      std::cout << (data.cpu_data())[i*(data.count()/10)] << " ";  
    std::cout << std::endl;
    
    // flow_forwardR
    binstream.read(reinterpret_cast<char *>(&flag), 4);
    std::cout << "Flag backwardR: " << flag << std::endl; 
    read_binstream(&binstream, BinaryDB_DataEncoding_FIXED16DIV32, data.count(), data.mutable_cpu_data());  
    flow_to_color<float>(shape.at(3), shape.at(2), data.cpu_data(), out_img.mutable_cpu_data(), flow_scale);  
    writePPM("/misc/lmbraid17/sceneflownet/common/sandbox/test_flow_backwardR.ppm", out_img.cpu_data(), shape.at(3), shape.at(2), true, 1.);  
    for(int i=0; i<10; i++)
      std::cout << (data.cpu_data())[i*(data.count()/10)] << " ";  
    std::cout << std::endl;
  } else 
    LOG(WARNING) << "Failed to open " << path;
  
  /// check reading disparities
  std::cout << "\n\n == Reading the disparity == \n";  
  
  float disp_scale = 0.02;
    
  binstream.close();
  path = data_path + "/disparity.bin";
  binstream.open(path.c_str(), 
                 std::ios::in | std::ios::binary);
  
  if (binstream.is_open()) {   
  
    shape.at(0) = 1;
    shape.at(1) = 1;
    shape.at(2) = 540;
    shape.at(3) = 960; 
    data.Reshape(shape); 
    
    offset = (long int)(data.count()*2+4)*nimg*2;
    std::cout << "\n Offset = " << offset << "\n";
    binstream.seekg(offset, ios::beg);
    
    // dispL
    binstream.read(reinterpret_cast<char *>(&flag), 4);
    std::cout << "Flag dispL: " << flag << std::endl; 
    read_binstream(&binstream, BinaryDB_DataEncoding_FIXED16DIV32, data.count(), data.mutable_cpu_data());  
    disp_to_color<float>(shape.at(3), shape.at(2), data.cpu_data(), out_img.mutable_cpu_data(), disp_scale);  
    writePPM("/misc/lmbraid17/sceneflownet/common/sandbox/test_dispL.ppm", out_img.cpu_data(), shape.at(3), shape.at(2), true, 1.);  
    for(int i=0; i<10; i++)
      std::cout << (data.cpu_data())[i*(data.count()/10)] << " ";  
    std::cout << std::endl;
    
    // dispR
    binstream.read(reinterpret_cast<char *>(&flag), 4);
    std::cout << "Flag dispR: " << flag << std::endl; 
    read_binstream(&binstream, BinaryDB_DataEncoding_FIXED16DIV32, data.count(), data.mutable_cpu_data());  
    disp_to_color<float>(shape.at(3), shape.at(2), data.cpu_data(), out_img.mutable_cpu_data(), disp_scale);  
    writePPM("/misc/lmbraid17/sceneflownet/common/sandbox/test_dispR.ppm", out_img.cpu_data(), shape.at(3), shape.at(2), true, 1.);  
    for(int i=0; i<10; i++)
      std::cout << (data.cpu_data())[i*(data.count()/10)] << " ";  
    std::cout << std::endl;
  } else 
    LOG(WARNING) << "Failed to open " << path;
  
  /// check reading disparity changes
  std::cout << "\n\n == Reading the disparity change == \n";  
  
  float disp_change_scale = 10;
    
  binstream.close();
  path = data_path + "/disparity_change.bin";
  binstream.open(path.c_str(), 
                 std::ios::in | std::ios::binary);
  
  if (binstream.is_open()) {   
  
    shape.at(0) = 1;
    shape.at(1) = 1;
    shape.at(2) = 540;
    shape.at(3) = 960; 
    data.Reshape(shape); 
    
    offset = (long int)(data.count()*2+4)*nimg*4;
    std::cout << "\n Offset = " << offset << "\n";
    binstream.seekg(offset, ios::beg);
    
    // disp_change_forwardL
    binstream.read(reinterpret_cast<char *>(&flag), 4);
    std::cout << "Flag disp_change_forwardL: " << flag << std::endl; 
    read_binstream(&binstream, BinaryDB_DataEncoding_FIXED16DIV32, data.count(), data.mutable_cpu_data());  
    disp_to_color<float>(shape.at(3), shape.at(2), data.cpu_data(), out_img.mutable_cpu_data(), disp_change_scale);  
    writePPM("/misc/lmbraid17/sceneflownet/common/sandbox/test_disp_change_forwardL.ppm", out_img.cpu_data(), shape.at(3), shape.at(2), true, 1.);  
    for(int i=0; i<10; i++)
      std::cout << (data.cpu_data())[i*(data.count()/10)] << " ";  
    std::cout << std::endl;
    
    // disp_change_forwardR
    binstream.read(reinterpret_cast<char *>(&flag), 4);
    std::cout << "Flag disp_change_forwardR: " << flag << std::endl; 
    read_binstream(&binstream, BinaryDB_DataEncoding_FIXED16DIV32, data.count(), data.mutable_cpu_data());  
    disp_to_color<float>(shape.at(3), shape.at(2), data.cpu_data(), out_img.mutable_cpu_data(), disp_change_scale);  
    writePPM("/misc/lmbraid17/sceneflownet/common/sandbox/test_disp_change_forwardR.ppm", out_img.cpu_data(), shape.at(3), shape.at(2), true, 1.);  
    for(int i=0; i<10; i++)
      std::cout << (data.cpu_data())[i*(data.count()/10)] << " ";  
    std::cout << std::endl;
    
    // disp_change_backwardL
    binstream.read(reinterpret_cast<char *>(&flag), 4);
    std::cout << "Flag disp_change_backwardL: " << flag << std::endl; 
    read_binstream(&binstream, BinaryDB_DataEncoding_FIXED16DIV32, data.count(), data.mutable_cpu_data());  
    disp_to_color<float>(shape.at(3), shape.at(2), data.cpu_data(), out_img.mutable_cpu_data(), disp_change_scale);  
    writePPM("/misc/lmbraid17/sceneflownet/common/sandbox/test_disp_change_backwardL.ppm", out_img.cpu_data(), shape.at(3), shape.at(2), true, 1.);  
    for(int i=0; i<10; i++)
      std::cout << (data.cpu_data())[i*(data.count()/10)] << " ";  
    std::cout << std::endl;
    
    // disp_change_backwardR
    binstream.read(reinterpret_cast<char *>(&flag), 4);
    std::cout << "Flag disp_change_backwardR: " << flag << std::endl; 
    read_binstream(&binstream, BinaryDB_DataEncoding_FIXED16DIV32, data.count(), data.mutable_cpu_data());  
    disp_to_color<float>(shape.at(3), shape.at(2), data.cpu_data(), out_img.mutable_cpu_data(), disp_change_scale);  
    writePPM("/misc/lmbraid17/sceneflownet/common/sandbox/test_disp_change_backwardR.ppm", out_img.cpu_data(), shape.at(3), shape.at(2), true, 1.);  
    for(int i=0; i<10; i++)
      std::cout << (data.cpu_data())[i*(data.count()/10)] << " ";  
    std::cout << std::endl;
  } else 
    LOG(WARNING) << "Failed to open " << path;
    
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

template <typename Dtype>
void flow_to_color(int width, int height, const Dtype* flow, Dtype* out_img, Dtype scale) {
  for (int x = 0; x< width; x++)
    for (int y=0; y<height; y++) {
      Dtype fx = flow[width*y + x] * scale;
      Dtype fy = flow[width*(y+height) + x] * scale;
      Dtype color[3];
      sintelCartesianToRGB(fx, fy, color);
      for(int c=0; c<3; c++)
        out_img[(c*height +y)*width + x] = color[c];
    }
      
}

template <typename Dtype>
void disp_to_color(int width, int height, const Dtype* disp, Dtype* out_img, Dtype scale) {
  for (int x = 0; x< width; x++)
    for (int y=0; y<height; y++) {
      Dtype fx = disp[width*y + x] * scale;
      Dtype fy = 0.;
      Dtype color[3];
      sintelCartesianToRGB(fx, fy, color);
      for(int c=0; c<3; c++)
        out_img[(c*height +y)*width + x] = color[c];
    }
      
}


static void setcols(int r, int g, int b, int k)
{
    colorwheel[k][0] = r;
    colorwheel[k][1] = g;
    colorwheel[k][2] = b;
}

static void makecolorwheel(void)
{
    // relative lengths of color transitions:
    // these are chosen based on perceptual similarity
    // (e.g. one can distinguish more shades between red and yellow 
    //  than between yellow and green)
    int RY = 15;
    int YG = 6;
    int GC = 4;
    int CB = 11;
    int BM = 13;
    int MR = 6;
    ncols = RY + YG + GC + CB + BM + MR;
    //printf("ncols = %d\n", ncols);
    if (ncols > MAXCOLS)
  exit(1);
    int i;
    int k = 0;
    for (i = 0; i < RY; i++) setcols(255,    255*i/RY,   0,        k++);
    for (i = 0; i < YG; i++) setcols(255-255*i/YG, 255,    0,        k++);
    for (i = 0; i < GC; i++) setcols(0,      255,    255*i/GC,     k++);
    for (i = 0; i < CB; i++) setcols(0,      255-255*i/CB, 255,        k++);
    for (i = 0; i < BM; i++) setcols(255*i/BM,     0,    255,        k++);
    for (i = 0; i < MR; i++) setcols(255,    0,    255-255*i/MR, k++);
}

static void sintelCartesianToRGB(float fx, float fy, float* pix)
{
    if (ncols == 0) makecolorwheel();

    float rad = sqrt(fx * fx + fy * fy);
    float a = atan2(-fy, -fx) / M_PI;
    float fk = (a + 1.0) / 2.0 * (ncols-1);
    int k0 = (int)fk;
    int k1 = (k0 + 1) % ncols;
    float f = fk - k0;
    //f = 0; // uncomment to see original color wheel
    int b;
    for (b = 0; b < 3; b++) {
      float col0 = colorwheel[k0][b] / 255.0;
      float col1 = colorwheel[k1][b] / 255.0;
      float col = (1 - f) * col0 + f * col1;
      if (rad <= 1)
          col = 1 - rad * (1 - col); // increase saturation with radius
      //else
      //    col *= .75; // out of range

      pix[2 - b] = (int)(255.0 * col);
      //pix[b] = col;
    }
}