#include <algorithm>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include "caffe/util/output.hpp"

using std::min;
using std::max;

namespace caffe {

template <typename TypeParam>
class Correlation1DLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  Correlation1DLayerTest()
      : epsilon_(Dtype(1e-3)),
        blob_bottom_1_(new Blob<Dtype>()),
        blob_bottom_2_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>())
  {}

  virtual void SetUp()
  {
    Caffe::set_random_seed(1701);
    //blob_bottom_->Reshape(1, 60, 120, 120 );
    //blob_bottom_1_->Reshape(1, 51, 219, 254 );
    //blob_bottom_2_->Reshape(1, 51, 219, 254 );
    blob_bottom_1_->Reshape(1, 1, 3, 3 );
    blob_bottom_2_->Reshape(1, 1, 3, 3 );
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_1_);
    filler.Fill(this->blob_bottom_2_);

    /*Dtype* ptr1=blob_bottom_1_->mutable_cpu_data();
    for(int i=0; i<blob_bottom_1_->count(); i++)
        ptr1[i]=1;

    Dtype* ptr2=blob_bottom_2_->mutable_cpu_data();
    for(int i=0; i<blob_bottom_1_->count(); i++)
        ptr2[i]=2;*/

    blob_bottom_vec_.push_back(blob_bottom_1_);
    blob_bottom_vec_.push_back(blob_bottom_2_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~Correlation1DLayerTest()
  {
      delete blob_bottom_1_;
      delete blob_bottom_2_;
      delete blob_top_;
  }
  void runGradTest(const char name[], int n, int c, int h, int w, int k, int pad, int maxdisp, int s1, int s2, CorrelationParameter_CorrelationType corrtype);
  void runFwdTest(const char name[], int n, int c, int h, int w, int k, int pad, int maxdisp, int s1, int s2, CorrelationParameter_CorrelationType corrtype);
  
  void ReferenceCorrelationForward(const Blob<Dtype>& blob_bottom_1, const Blob<Dtype>& blob_bottom_2,
      const LayerParameter& layer_param, Blob<Dtype>* blob_top);

  Dtype epsilon_;
  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

template <typename TypeParam>
void Correlation1DLayerTest<TypeParam>::ReferenceCorrelationForward(
    const Blob<Dtype>& blob_bottom_1, const Blob<Dtype>& blob_bottom_2, const LayerParameter& layer_param,
    Blob<Dtype>* blob_top)
{
  CorrelationParameter corr_param = layer_param.correlation_param();

  CHECK(corr_param.has_kernel_size()) << "Filter kernel_size is not set";
  CHECK(corr_param.has_max_displacement()) << "Max displacement is required.";

  int kernel_size = corr_param.kernel_size();
  if(kernel_size % 2 == 0) LOG(FATAL) << "Odd kernel size required";

  int maxdisplacement = corr_param.max_displacement();
  int pad_size = corr_param.pad();
  int stride1 = corr_param.stride_1();
  int stride2 = corr_param.stride_2();

  //LOG(INFO) << "Pad Size: " << pad_size;  
  //LOG(INFO) << "Kernel Size: " << kernel_size;
  //LOG(INFO) << "Stride 1: " << stride1;
  //LOG(INFO) << "Stride 2: " << stride2;
  //LOG(INFO) << "Max Displacement: " << maxdisplacement;

  /**** Padding ****/
  const Dtype* bottom_data_1  = blob_bottom_1.cpu_data(); 
  const Dtype* bottom_data_2  = blob_bottom_2.cpu_data();
  
  // Create padded blobs
  Blob<Dtype> padded_blob_bottom_1;
  Blob<Dtype> padded_blob_bottom_2;

  padded_blob_bottom_1.Reshape(1,blob_bottom_1.channels(), blob_bottom_1.height() + 2 * pad_size, blob_bottom_1.width() + 2 * pad_size);
  padded_blob_bottom_2.Reshape(1,blob_bottom_1.channels(), blob_bottom_1.height() + 2 * pad_size, blob_bottom_1.width() + 2 * pad_size);

  //LOG(INFO) << "Padded Blob YSize: " << padded_blob_bottom_1.height();
  //LOG(INFO) << "Padded Blob XSize: " << padded_blob_bottom_1.width();
  //LOG(INFO) << "Padded Blob Chans: " << padded_blob_bottom_1.channels();
 
  Dtype* padded_blob_bottom_data_1=padded_blob_bottom_1.mutable_cpu_data();
  Dtype* padded_blob_bottom_data_2=padded_blob_bottom_2.mutable_cpu_data();
  
 
  // Initialize padded blobs with zero '0'
  for(int pC = 0; pC < blob_bottom_1.channels(); pC++){
    for(int pIndy = 0; pIndy < padded_blob_bottom_1.height() ; pIndy++ ){
      for(int pIndx = 0; pIndx < padded_blob_bottom_1.width(); pIndx++ ){
         padded_blob_bottom_data_1[padded_blob_bottom_1.offset(0, pC, pIndy, pIndx)] = 0.0;
         padded_blob_bottom_data_2[padded_blob_bottom_2.offset(0, pC, pIndy, pIndx)] = 0.0;
      }
      
    }
  }
 
  // Assign the blobs to padded blobs (centered on the padded blobs)
  for(int pC = 0; pC < blob_bottom_1.channels(); pC++){
    for(int pIndy = pad_size; pIndy < blob_bottom_1.height() + pad_size ; pIndy++ ){
      for(int pIndx = pad_size; pIndx < blob_bottom_1.width() + pad_size; pIndx++ ){
         padded_blob_bottom_data_1[padded_blob_bottom_1.offset(0, pC, pIndy, pIndx)] = 
                           bottom_data_1[blob_bottom_1.offset(0, pC, pIndy-pad_size, pIndx-pad_size)];
         padded_blob_bottom_data_2[padded_blob_bottom_2.offset(0, pC, pIndy, pIndx)] = 
                           bottom_data_2[blob_bottom_2.offset(0, pC, pIndy-pad_size, pIndx-pad_size)];
      }
    }
  }
  
 
  // Read size of the blob
  int numCols    =padded_blob_bottom_1.width();
  int numRows    =padded_blob_bottom_1.height();
  int numChannels=padded_blob_bottom_1.channels();

  // Write padded blobs to the files
  //writeFloatFile("test_correlation_blob1.float3", (const float*)padded_blob_bottom_data_1, numCols, numRows, numChannels );
  //writeFloatFile("test_correlation_blob2.float3", (const float*)padded_blob_bottom_data_2, numCols, numRows, numChannels );
  
  // Compute the size of top blob
  int top_numCols= 0;
  int top_numRows= 0;
  int top_numChannels= 0;

  // Offset -> where to start to compute the correlation
  int offset_ = floor( kernel_size / 2.0 ) + maxdisplacement;
  
  // Find the area in the second blob and divide it by the stride of the first blob
  top_numCols = ceil( (numCols - 2*offset_) / (float) stride1 ) ;
  top_numRows = ceil( (numRows - 2*offset_) / (float) stride1 ) ;
  int grid_radius = maxdisplacement / stride2;
  top_numChannels = (2* grid_radius+1);
  
  LOG(INFO) << "Top Blob XSize: " << top_numCols;
  LOG(INFO) << "Top Blob YSize: " << top_numRows;
  LOG(INFO) << "Top Blob Chans: " << top_numChannels;
 
  // Reshape top_blob to store the output
  blob_top->Reshape(1, top_numChannels, top_numRows, top_numCols);
  Dtype* top_data       = blob_top->mutable_cpu_data();
 
  // Initialize the top_data to zero
  for(int t=0; t< blob_top->count(); t++)
    top_data[t] = 0;

  int top_y= 0, top_x = 0, top_c = 0;
 
  // For each pixel in the 'do-able' area on the second blob
  // Indices increased by the the stride of first blob
  for(int y= offset_ ; y < numRows - offset_ ; y += stride1, top_y++ ){
    top_x = 0;
    for(int x= offset_ ; x < numCols - offset_ ; x += stride1, top_x++ ){

      // For each channel
      for(int c=0 ; c< numChannels; c++){

        top_c = 0;
        // For each displacement
        // multiply the two blobs and sum the value to the blob_top
        { int y2 = y; //for(int y2 = y - maxdisplacement + (maxdisplacement % stride2); y2 <= y +  maxdisplacement - (maxdisplacement % stride2);  y2 = y2 + stride2){
          for(int x2 = x - maxdisplacement+ (maxdisplacement % stride2); x2 <= x + maxdisplacement - (maxdisplacement % stride2);  x2 = x2 + stride2)
          {
             // Element-wise multiplication of patches (pathc is determined by kernel)
             for(int ky= - floor(kernel_size) / 2 ; ky <= floor(kernel_size) / 2; ky++)    // ky : kernel_y
               for(int kx= - floor(kernel_size) / 2 ; kx <= floor(kernel_size) / 2; kx++){ // kx : kernel_x
                   if(corr_param.correlation_type() == CorrelationParameter_CorrelationType_MULTIPLY) {
                        top_data[blob_top->offset(0, top_c, top_y, top_x)] += padded_blob_bottom_data_1[padded_blob_bottom_1.offset(0, c, y+ky, x+kx)] *
                                                                        padded_blob_bottom_data_2[padded_blob_bottom_2.offset(0, c, y2+ky, x2+kx)];
                   } else if(corr_param.correlation_type() == CorrelationParameter_CorrelationType_SUBTRACT) {
                        top_data[blob_top->offset(0, top_c, top_y, top_x)] += fabs(padded_blob_bottom_data_1[padded_blob_bottom_1.offset(0, c, y+ky, x+kx)] -
                                                                        padded_blob_bottom_data_2[padded_blob_bottom_2.offset(0, c, y2+ky, x2+kx)]);
                   }
               }
             top_c ++;
          }
        }
      }
    }
    
  }

  for(int t=0; t< blob_top->count(); t++) top_data[t] /= kernel_size*kernel_size*numChannels;

}

TYPED_TEST_CASE(Correlation1DLayerTest, TestDtypesAndDevices);
//TYPED_TEST_CASE(Correlation1DLayerTest, ::testing::Types<FloatGPU>);

template <typename TypeParam>
void Correlation1DLayerTest<TypeParam>::runFwdTest(const char name[], int n, int c, int h, int w, int k, int pad, int maxdisp, int s1, int s2, CorrelationParameter_CorrelationType corrtype)
{
    typedef typename TypeParam::Dtype Dtype;
    
    LOG(INFO) << "Running FORWARD TEST >" << name << "< with following params:";
    LOG(INFO) << "Bottom blobs sizes: [n,c,h,w] = " << n << "," << c << "," << h << "," << w;
    LOG(INFO) << "Kernel size: " << k << " Padding: " << pad;
    LOG(INFO) << "Max displacement: " << maxdisp << " Stride1: " << s1 << " Stride2: " << s2;
    
    blob_bottom_1_->Reshape(n, c, h, w );
    blob_bottom_2_->Reshape(n, c, h, w );
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_1_);
    filler.Fill(this->blob_bottom_2_);

    // Fixed Data:
    /*Dtype* ptr1=blob_bottom_1_->mutable_cpu_data();
    for(int i=0; i<blob_bottom_1_->count(); i++)
        ptr1[i]=1;

    Dtype* ptr2=blob_bottom_2_->mutable_cpu_data();
    for(int i=0; i<blob_bottom_1_->count(); i++)
        ptr2[i]=2;*/
        
    LayerParameter layer_param;

    CorrelationParameter* corr_param =
        layer_param.mutable_correlation_param();
        
    corr_param->set_correlation_type(corrtype);
    corr_param->set_kernel_size(k);
    corr_param->set_pad(pad);
    corr_param->set_max_displacement(maxdisp);
    corr_param->set_stride_1(s1);
    corr_param->set_stride_2(s2);
    
    Correlation1DLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    Blob<Dtype> top_reference;
    this->ReferenceCorrelationForward(*(this->blob_bottom_1_),*(this->blob_bottom_2_), layer_param,
        &top_reference);
    for (int i = 0; i < this->blob_top_->count(); ++i) {
        EXPECT_NEAR(this->blob_top_->cpu_data()[i], top_reference.cpu_data()[i], this->epsilon_);
    }
        
}

TYPED_TEST(Correlation1DLayerTest, TestForward) {
    if(Caffe::mode()==Caffe::CPU)
    {
        LOG(INFO) << "Skipping CPU test";
        return;
    }

    {
        CorrelationParameter_CorrelationType ctype = CorrelationParameter_CorrelationType_MULTIPLY;
        
        //name, n, c, h, w,   k, pad, maxdisp, s1, s2
        this->runFwdTest("Minimal", 1, 1, 1, 1,  1, 0, 0, 1, 1, ctype);
        
        this->runFwdTest("PaddingA", 1, 1, 1, 1,  1, 1, 0, 1, 1, ctype);
        this->runFwdTest("PaddingB", 1, 1, 1, 1,  1, 2, 0, 1, 1, ctype);
        this->runFwdTest("PaddingC", 1, 1, 1, 1,  3, 1, 0, 1, 1, ctype);

        this->runFwdTest("UselessStride1A", 1, 1, 1, 1,  3, 1, 0, 2, 1, ctype);
        this->runFwdTest("UselessStride1B", 1, 1, 1, 1,  3, 1, 0, 3, 1, ctype);
        this->runFwdTest("UselessStride1C", 1, 1, 1, 1,  3, 1, 0, 4, 1, ctype);
        this->runFwdTest("UselessStride2A", 1, 1, 1, 1,  3, 1, 0, 1, 2, ctype);
        this->runFwdTest("UselessStride2B", 1, 1, 1, 1,  3, 1, 0, 1, 3, ctype);
        this->runFwdTest("UselessStride2C", 1, 1, 1, 1,  3, 1, 0, 1, 4, ctype);
        this->runFwdTest("UselessStride3", 1, 1, 1, 1,  3, 1, 0, 2, 3, ctype);

        this->runFwdTest("Stride1A", 1, 1, 3, 3,  1, 1, 0, 2, 1, ctype);
        this->runFwdTest("Stride1B", 1, 1, 3, 3,  1, 0, 0, 2, 1, ctype);

        this->runFwdTest("ChannelsA", 1, 3, 3, 3,  1, 1, 1, 2, 1, ctype);
        this->runFwdTest("ChannelsB", 1, 3, 1, 1,  1, 1, 1, 1, 1, ctype);

        this->runFwdTest("BigA", 1, 1, 9, 16,  3, 1, 4, 1, 1, ctype);
        this->runFwdTest("BigKernelA", 1, 1, 9, 16,  5, 1, 3, 1, 1, ctype);
        this->runFwdTest("BigImageA", 1, 3, 34, 24,  7, 2, 5, 2, 3, ctype);
    }
    {
        CorrelationParameter_CorrelationType ctype = CorrelationParameter_CorrelationType_SUBTRACT;
        
        //name, n, c, h, w,   k, pad, maxdisp, s1, s2
        this->runFwdTest("Minimal", 1, 1, 1, 1,  1, 0, 0, 1, 1, ctype);
        
        this->runFwdTest("PaddingA", 1, 1, 1, 1,  1, 1, 0, 1, 1, ctype);
        this->runFwdTest("PaddingB", 1, 1, 1, 1,  1, 2, 0, 1, 1, ctype);
        this->runFwdTest("PaddingC", 1, 1, 1, 1,  3, 1, 0, 1, 1, ctype);

        this->runFwdTest("UselessStride1A", 1, 1, 1, 1,  3, 1, 0, 2, 1, ctype);
        this->runFwdTest("UselessStride1B", 1, 1, 1, 1,  3, 1, 0, 3, 1, ctype);
        this->runFwdTest("UselessStride1C", 1, 1, 1, 1,  3, 1, 0, 4, 1, ctype);
        this->runFwdTest("UselessStride2A", 1, 1, 1, 1,  3, 1, 0, 1, 2, ctype);
        this->runFwdTest("UselessStride2B", 1, 1, 1, 1,  3, 1, 0, 1, 3, ctype);
        this->runFwdTest("UselessStride2C", 1, 1, 1, 1,  3, 1, 0, 1, 4, ctype);
        this->runFwdTest("UselessStride3", 1, 1, 1, 1,  3, 1, 0, 2, 3, ctype);

        this->runFwdTest("Stride1A", 1, 1, 3, 3,  1, 1, 0, 2, 1, ctype);
        this->runFwdTest("Stride1B", 1, 1, 3, 3,  1, 0, 0, 2, 1, ctype);

        this->runFwdTest("ChannelsA", 1, 3, 3, 3,  1, 1, 1, 2, 1, ctype);
        this->runFwdTest("ChannelsB", 1, 3, 1, 1,  1, 1, 1, 1, 1, ctype);

        this->runFwdTest("BigA", 1, 1, 9, 16,  3, 1, 4, 1, 1, ctype);
        this->runFwdTest("BigKernelA", 1, 1, 9, 16,  5, 1, 3, 1, 1, ctype);
        this->runFwdTest("BigImageA", 1, 3, 34, 24,  7, 2, 5, 2, 3, ctype);
    }
}


template <typename TypeParam>
void Correlation1DLayerTest<TypeParam>::runGradTest(const char name[], int n, int c, int h, int w, int k, int pad, int maxdisp, int s1, int s2, CorrelationParameter_CorrelationType corrtype)
{
    typedef typename TypeParam::Dtype Dtype;
    
    LOG(INFO) << "Running GRADIENT TEST >" << name << "< with following params:";
    LOG(INFO) << "Bottom blobs sizes: [n,c,h,w] = " << n << "," << c << "," << h << "," << w;
    LOG(INFO) << "Kernel size: " << k << " Padding: " << pad;
    LOG(INFO) << "Max displacement: " << maxdisp << " Stride1: " << s1 << " Stride2: " << s2;
    
    blob_bottom_1_->Reshape(n, c, h, w );
    blob_bottom_2_->Reshape(n, c, h, w );
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_1_);
    filler.Fill(this->blob_bottom_2_);

    // Fixed Data:
    /*Dtype* ptr1=blob_bottom_1_->mutable_cpu_data();
    for(int i=0; i<blob_bottom_1_->count(); i++)
        ptr1[i]=1;

    Dtype* ptr2=blob_bottom_2_->mutable_cpu_data();
    for(int i=0; i<blob_bottom_1_->count(); i++)
        ptr2[i]=2;*/
        
    LayerParameter layer_param;

    CorrelationParameter* corr_param =
        layer_param.mutable_correlation_param();
        
    corr_param->set_kernel_size(k);
    corr_param->set_pad(pad);
    corr_param->set_max_displacement(maxdisp);
    corr_param->set_stride_1(s1);
    corr_param->set_stride_2(s2);
    
    Correlation1DLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-2);

    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    for (int i = 0; i < this->blob_top_->count(); ++i) {
      this->blob_top_->mutable_cpu_diff()[i] = 1.;
    }
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
    
}

TYPED_TEST(Correlation1DLayerTest, TestGradient) {
    if(Caffe::mode()==Caffe::CPU)
    {
        LOG(INFO) << "Skipping CPU test";
        return;
    }

    //this->runGradTest("UselessStride1B", 1, 1, 1, 1,  3, 1, 0, 3, 1);
    
    {
        CorrelationParameter_CorrelationType ctype = CorrelationParameter_CorrelationType_MULTIPLY;
        
        //name, n, c, h, w,   k, pad, maxdisp, s1, s2
        this->runGradTest("Minimal", 1, 1, 1, 1,  1, 0, 0, 1, 1, ctype);
        
        this->runGradTest("PaddingA", 1, 1, 1, 1,  1, 1, 0, 1, 1, ctype);
        this->runGradTest("PaddingB", 1, 1, 1, 1,  1, 2, 0, 1, 1, ctype);
        this->runGradTest("PaddingC", 1, 1, 1, 1,  3, 1, 0, 1, 1, ctype);

        this->runGradTest("UselessStride1A", 1, 1, 1, 1,  3, 1, 0, 2, 1, ctype);
        this->runGradTest("UselessStride1B", 1, 1, 1, 1,  3, 1, 0, 3, 1, ctype);
        this->runGradTest("UselessStride1C", 1, 1, 1, 1,  3, 1, 0, 4, 1, ctype);
        this->runGradTest("UselessStride2A", 1, 1, 1, 1,  3, 1, 0, 1, 2, ctype);
        this->runGradTest("UselessStride2B", 1, 1, 1, 1,  3, 1, 0, 1, 3, ctype);
        this->runGradTest("UselessStride2C", 1, 1, 1, 1,  3, 1, 0, 1, 4, ctype);
        this->runGradTest("UselessStride3", 1, 1, 1, 1,  3, 1, 0, 2, 3, ctype);

        this->runGradTest("Stride1A", 1, 1, 3, 3,  1, 1, 0, 2, 1, ctype);
        this->runGradTest("Stride1B", 1, 1, 3, 3,  1, 0, 0, 2, 1, ctype);

        this->runGradTest("ChannelsA", 1, 3, 3, 3,  1, 1, 1, 2, 1, ctype);
        this->runGradTest("ChannelsB", 1, 3, 1, 1,  1, 1, 1, 1, 1, ctype);

        this->runGradTest("BigA", 1, 1, 9, 16,  3, 1, 4, 1, 1, ctype);
    }
    {
        CorrelationParameter_CorrelationType ctype = CorrelationParameter_CorrelationType_SUBTRACT;
        
        //name, n, c, h, w,   k, pad, maxdisp, s1, s2
        this->runGradTest("Minimal", 1, 1, 1, 1,  1, 0, 0, 1, 1, ctype);
        
        this->runGradTest("PaddingA", 1, 1, 1, 1,  1, 1, 0, 1, 1, ctype);
        this->runGradTest("PaddingB", 1, 1, 1, 1,  1, 2, 0, 1, 1, ctype);
        this->runGradTest("PaddingC", 1, 1, 1, 1,  3, 1, 0, 1, 1, ctype);

        this->runGradTest("UselessStride1A", 1, 1, 1, 1,  3, 1, 0, 2, 1, ctype);
        this->runGradTest("UselessStride1B", 1, 1, 1, 1,  3, 1, 0, 3, 1, ctype);
        this->runGradTest("UselessStride1C", 1, 1, 1, 1,  3, 1, 0, 4, 1, ctype);
        this->runGradTest("UselessStride2A", 1, 1, 1, 1,  3, 1, 0, 1, 2, ctype);
        this->runGradTest("UselessStride2B", 1, 1, 1, 1,  3, 1, 0, 1, 3, ctype);
        this->runGradTest("UselessStride2C", 1, 1, 1, 1,  3, 1, 0, 1, 4, ctype);
        this->runGradTest("UselessStride3", 1, 1, 1, 1,  3, 1, 0, 2, 3, ctype);

        this->runGradTest("Stride1A", 1, 1, 3, 3,  1, 1, 0, 2, 1, ctype);
        this->runGradTest("Stride1B", 1, 1, 3, 3,  1, 0, 0, 2, 1, ctype);

        this->runGradTest("ChannelsA", 1, 3, 3, 3,  1, 1, 1, 2, 1, ctype);
        this->runGradTest("ChannelsB", 1, 3, 1, 1,  1, 1, 1, 1, 1, ctype);

        this->runGradTest("BigA", 1, 1, 9, 16,  3, 1, 4, 1, 1, ctype);
    }

}


}  // namespace caffe
