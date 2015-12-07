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
class ToleranceLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ToleranceLayerTest()
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

  virtual ~ToleranceLayerTest()
  {
      delete blob_bottom_1_;
      delete blob_bottom_2_;
      delete blob_top_;
  }
  void runFwdTest(const char name[], int n, int c, int h, int w, int radius);
  
  void ReferenceToleranceForward(const Blob<Dtype>& blob_bottom_1, const Blob<Dtype>& blob_bottom_2,
      const LayerParameter& layer_param, Blob<Dtype>* blob_top);

  Dtype epsilon_;
  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

template <typename TypeParam>
void ToleranceLayerTest<TypeParam>::ReferenceToleranceForward(
    const Blob<Dtype>& blob_bottom_1, const Blob<Dtype>& blob_bottom_2, const LayerParameter& layer_param,
    Blob<Dtype>* blob_top)
{
  ToleranceParameter tolerance_param = layer_param.tolerance_param();

  int rad = tolerance_param.tolerance_radius();
  
  // Reshape top_blob to store the output
  blob_top->ReshapeLike(blob_bottom_1);
  
  Dtype* top_data       = blob_top->mutable_cpu_data();
  const Dtype* gt_data       = blob_bottom_1.cpu_data();
  const Dtype* data       = blob_bottom_2.cpu_data();
 
  // Initialize the top_data to special number
  for(int t=0; t< blob_top->count(); t++) top_data[t] = 123.456f;

  
  int width = blob_bottom_1.width();
  int height = blob_bottom_1.height();
  int channels = blob_bottom_1.channels();
  int num = blob_bottom_1.num();
  
  for(int n=0; n < num; n++) {
    for(int c=0; c < channels; c++) {
      
      for(int h=0; h < height; h++) {
        for(int w=0; w < width; w++) {
          
          float data_val = data[blob_bottom_2.offset(n,c,h,w)];
          
          // Now find closest GT val in neighborhood
          int init = 1;
          float min_diff = 0;
          float min_diff_gt_val = 0;
          for(int i=-rad; i <= rad; i++) {
            for(int j=-rad; j <= rad; j++) {
              int xpos = w+i;
              int ypos = h+j;
              if(xpos < 0 || ypos < 0 || xpos >= width || ypos >= height) continue;
              
              float gt_val = gt_data[blob_bottom_1.offset(n,c,ypos,xpos)];
              
              float diff = fabsf(gt_val - data_val);
              //printf("RefImpl: [%d,%d] compare %f < %f -> %f?\n", w,h,diff,min_diff,gt_val);
              if(diff < min_diff || init) {
                init = 0;
                min_diff = diff;
                min_diff_gt_val = gt_val;
              }
            }
          }
          
          top_data[blob_top->offset(n,c,h,w)] = min_diff_gt_val;
        } 
      }      
    }
  }
  
  // Completeness check
  for(int t=0; t< blob_top->count(); t++) if(top_data[t] == 123.456f) LOG(FATAL) << "Unset element at index " << t;
}

TYPED_TEST_CASE(ToleranceLayerTest, TestDtypesAndDevices);

template <typename TypeParam>
void ToleranceLayerTest<TypeParam>::runFwdTest(const char name[], int n, int c, int h, int w, int radius)
{
    typedef typename TypeParam::Dtype Dtype;
    
    LOG(INFO) << "Running FORWARD TEST >" << name << "< with following params:";
    LOG(INFO) << "Bottom blobs sizes: [n,c,h,w] = " << n << "," << c << "," << h << "," << w;
    LOG(INFO) << "Radius: " << radius;
    
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

    ToleranceParameter* tol_param =
        layer_param.mutable_tolerance_param();
        
    tol_param->set_tolerance_radius(radius);
    
    ToleranceLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    Blob<Dtype> top_reference;
    this->ReferenceToleranceForward(*(this->blob_bottom_1_),*(this->blob_bottom_2_), layer_param,
        &top_reference);
    LOG(INFO) << "W (impl vs. reference) = " << this->blob_top_->width() << " vs. " << top_reference.width();
    LOG(INFO) << "H (impl vs. reference) = " << this->blob_top_->height() << " vs. " << top_reference.height();
    LOG(INFO) << "C (impl vs. reference) = " << this->blob_top_->channels() << " vs. " << top_reference.channels();
    
    CHECK_EQ(this->blob_top_->width(), top_reference.width());
    CHECK_EQ(this->blob_top_->height(), top_reference.height());
    CHECK_EQ(this->blob_top_->channels(), top_reference.channels());
    
    /*this->blob_bottom_1_->print("BOT0");
    this->blob_bottom_2_->print("BOT1");
    blob_top_->print("IMPL");
    top_reference.print("REF");*/
    
    std::cout.flush();
    
    for (int i = 0; i < this->blob_top_->count(); ++i) {
        //printf("[%d]", i);
        //std::cout.flush();
        EXPECT_NEAR(this->blob_top_->cpu_data()[i], top_reference.cpu_data()[i], this->epsilon_);
    }
        
}

TYPED_TEST(ToleranceLayerTest, TestForward) {
    if(Caffe::mode()==Caffe::CPU)
    {
        LOG(INFO) << "Skipping CPU test";
        return;
    }
    
    //name, n, c, h, w,   radius
    this->runFwdTest("MinimalUseless", 1, 1, 1, 1,  1);
    this->runFwdTest("MinimalUseful", 1, 1, 1, 2,  1);
    
    this->runFwdTest("Small", 1, 1, 3, 3,  1);
    this->runFwdTest("Medium", 1, 1, 5, 5,  3);
    this->runFwdTest("BigA", 1, 1, 50, 50,  1);
    this->runFwdTest("BigB", 1, 1, 50, 50,  3);
    this->runFwdTest("BigC", 1, 1, 50, 50,  5);

    this->runFwdTest("Channels-n-Batches", 5, 5, 10, 10,  2);
    
    this->runFwdTest("BigSlimH", 5, 5, 3, 100,  2);
    this->runFwdTest("BigSlimV", 5, 5, 100, 3,  2);
    
    this->runFwdTest("BigNeighborhood", 1, 1, 100, 100,  30);
    
    
}



}  // namespace caffe
