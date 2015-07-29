#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class UnpoolingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  UnpoolingLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_bottom_mask_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(2, 3, 6, 5);
    blob_bottom_mask_->Reshape(2, 3, 6, 5);
    // fill the values for the pooled map
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    // fill the values for the mask
    FillerParameter ufiller_param;
    ufiller_param.set_max(29);
    UniformFiller<Dtype> uniform_filler(ufiller_param);
    uniform_filler.Fill(this->blob_bottom_mask_);
    
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_mask_);
    blob_top_vec_.push_back(blob_top_);
    propagate_down_.push_back(true);
  }
  virtual ~UnpoolingLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
    delete blob_bottom_mask_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_mask_;  
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  vector<bool> propagate_down_;
  
  // Test for 2x 2 square pooling layer
  void TestForwardSquare() {
    LayerParameter layer_param;
    UnpoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
    unpooling_param->set_kernel_size(2);
    const int num = 2;
    const int channels = 2;
    blob_bottom_->Reshape(num, channels, 2, 4);
    blob_bottom_mask_->Reshape(num, channels, 2, 4);
    
    // In (pooled map): 2x 2 channels of:
    //     [9 5 5 8]
    //     [9 5 5 8]
     for (int i = 0; i < 8 * num * channels; i += 8) {
      blob_bottom_->mutable_cpu_data()[i + 0] = 9;
      blob_bottom_->mutable_cpu_data()[i + 1] = 5;
      blob_bottom_->mutable_cpu_data()[i + 2] = 5;
      blob_bottom_->mutable_cpu_data()[i + 3] = 8;
      blob_bottom_->mutable_cpu_data()[i + 4] = 9;
      blob_bottom_->mutable_cpu_data()[i + 5] = 5;
      blob_bottom_->mutable_cpu_data()[i + 6] = 5;
      blob_bottom_->mutable_cpu_data()[i + 7] = 8;
    }
    
    // In (mask): 2x 2 channels of:
    //     [5  2  2 9]
    //     [5 12 12 9]
     for (int i = 0; i < 8 * num * channels; i += 8) {
      blob_bottom_mask_->mutable_cpu_data()[i + 0] =  5;
      blob_bottom_mask_->mutable_cpu_data()[i + 1] =  2;
      blob_bottom_mask_->mutable_cpu_data()[i + 2] =  2;
      blob_bottom_mask_->mutable_cpu_data()[i + 3] =  9;
      blob_bottom_mask_->mutable_cpu_data()[i + 4] =  5;
      blob_bottom_mask_->mutable_cpu_data()[i + 5] = 12;
      blob_bottom_mask_->mutable_cpu_data()[i + 6] = 12;
      blob_bottom_mask_->mutable_cpu_data()[i + 7] =  9;
    }     
    
    // set up layer
    UnpoolingLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    
    // check top shape
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), 3);
    EXPECT_EQ(blob_top_->width(), 5);
    
    // forward pass and check output
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    
    // Expected output: 2x 2 channels of:
    //     [0 0 5 0 0]
    //     [9 0 0 0 8]
    //     [0 0 5 0 0]
    for (int i = 0; i < 15 * num * channels; i += 15) {
      EXPECT_EQ(blob_top_->cpu_data()[i +  0], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i +  1], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i +  2], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i +  3], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i +  4], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i +  5], 9);
      EXPECT_EQ(blob_top_->cpu_data()[i +  6], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i +  7], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i +  8], 0);      
      EXPECT_EQ(blob_top_->cpu_data()[i +  9], 8);      
      EXPECT_EQ(blob_top_->cpu_data()[i + 10], 0);      
      EXPECT_EQ(blob_top_->cpu_data()[i + 11], 0);      
      EXPECT_EQ(blob_top_->cpu_data()[i + 12], 5);      
      EXPECT_EQ(blob_top_->cpu_data()[i + 13], 0);      
      EXPECT_EQ(blob_top_->cpu_data()[i + 14], 0);      
    }    
  }

  // Test for 3x 2 rectangular pooling layer with kernel_h > kernel_w
  void TestForwardRectHigh() {
    LayerParameter layer_param;
    UnpoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
    unpooling_param->set_kernel_h(3);
    unpooling_param->set_kernel_w(2);
    const int num = 2;
    const int channels = 2;
    blob_bottom_->Reshape(num, channels, 4, 5);
    blob_bottom_mask_->Reshape(num, channels, 4, 5);
    
    // In (pooled map): 2x 2 channels of:
    // [35    32    26    27    27]
    // [32    33    33    27    27]
    // [31    34    34    27    27]
    // [36    36    34    18    18]
    for (int i = 0; i < 20 * num * channels; i += 20) {
      blob_bottom_->mutable_cpu_data()[i +  0] = 35;
      blob_bottom_->mutable_cpu_data()[i +  1] = 32;
      blob_bottom_->mutable_cpu_data()[i +  2] = 26;
      blob_bottom_->mutable_cpu_data()[i +  3] = 27;
      blob_bottom_->mutable_cpu_data()[i +  4] = 27;
      blob_bottom_->mutable_cpu_data()[i +  5] = 32;
      blob_bottom_->mutable_cpu_data()[i +  6] = 33;
      blob_bottom_->mutable_cpu_data()[i +  7] = 33;
      blob_bottom_->mutable_cpu_data()[i +  8] = 27;
      blob_bottom_->mutable_cpu_data()[i +  9] = 27;
      blob_bottom_->mutable_cpu_data()[i + 10] = 31;
      blob_bottom_->mutable_cpu_data()[i + 11] = 34;
      blob_bottom_->mutable_cpu_data()[i + 12] = 34;
      blob_bottom_->mutable_cpu_data()[i + 13] = 27;
      blob_bottom_->mutable_cpu_data()[i + 14] = 27;
      blob_bottom_->mutable_cpu_data()[i + 15] = 36;
      blob_bottom_->mutable_cpu_data()[i + 16] = 36;
      blob_bottom_->mutable_cpu_data()[i + 17] = 34;
      blob_bottom_->mutable_cpu_data()[i + 18] = 18;
      blob_bottom_->mutable_cpu_data()[i + 19] = 18;
    }

    // In (mask): 2x 2 channels of:
    // [ 0     7     3    16    16]
    // [ 7    20    20    16    16]
    // [12    26    26    16    16]
    // [31    31    26    34    34]
    for (int i = 0; i < 20 * num * channels; i += 20) {
      blob_bottom_mask_->mutable_cpu_data()[i +  0] =  0;
      blob_bottom_mask_->mutable_cpu_data()[i +  1] =  7;
      blob_bottom_mask_->mutable_cpu_data()[i +  2] =  3;
      blob_bottom_mask_->mutable_cpu_data()[i +  3] = 16;
      blob_bottom_mask_->mutable_cpu_data()[i +  4] = 16;
      blob_bottom_mask_->mutable_cpu_data()[i +  5] =  7;
      blob_bottom_mask_->mutable_cpu_data()[i +  6] = 20;
      blob_bottom_mask_->mutable_cpu_data()[i +  7] = 20;
      blob_bottom_mask_->mutable_cpu_data()[i +  8] = 16;
      blob_bottom_mask_->mutable_cpu_data()[i +  9] = 16;
      blob_bottom_mask_->mutable_cpu_data()[i + 10] = 12;
      blob_bottom_mask_->mutable_cpu_data()[i + 11] = 26;
      blob_bottom_mask_->mutable_cpu_data()[i + 12] = 26;
      blob_bottom_mask_->mutable_cpu_data()[i + 13] = 16;
      blob_bottom_mask_->mutable_cpu_data()[i + 14] = 16;
      blob_bottom_mask_->mutable_cpu_data()[i + 15] = 31;
      blob_bottom_mask_->mutable_cpu_data()[i + 16] = 31;
      blob_bottom_mask_->mutable_cpu_data()[i + 17] = 26;
      blob_bottom_mask_->mutable_cpu_data()[i + 18] = 34;
      blob_bottom_mask_->mutable_cpu_data()[i + 19] = 34;
    }
   
    UnpoolingLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), 6);
    EXPECT_EQ(blob_top_->width(), 6);
    
    layer.Forward(blob_bottom_vec_, blob_top_vec_);

    // Expected outp:
    // [35     0     0    26     0     0]
    // [ 0    32     0     0     0     0]
    // [31     0     0     0    27     0]
    // [ 0     0    33     0     0     0]
    // [ 0     0    34     0     0     0]
    // [ 0    36     0     0    18     0]    
    for (int i = 0; i < 36 * num * channels; i += 36) {
      EXPECT_EQ(blob_top_->cpu_data()[i +  0], 35);
      EXPECT_EQ(blob_top_->cpu_data()[i +  1], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i +  2], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i +  3], 26);
      EXPECT_EQ(blob_top_->cpu_data()[i +  4], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i +  5], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i +  6], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i +  7], 32);
      EXPECT_EQ(blob_top_->cpu_data()[i +  8], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i +  9], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 10], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 11], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 12], 31);
      EXPECT_EQ(blob_top_->cpu_data()[i + 13], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 14], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 15], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 16], 27);
      EXPECT_EQ(blob_top_->cpu_data()[i + 17], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 18], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 19], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 20], 33);
      EXPECT_EQ(blob_top_->cpu_data()[i + 21], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 22], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 23], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 24], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 25], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 26], 34);
      EXPECT_EQ(blob_top_->cpu_data()[i + 27], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 28], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 29], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 30], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 31], 36);
      EXPECT_EQ(blob_top_->cpu_data()[i + 32], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 33], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 34], 18);
      EXPECT_EQ(blob_top_->cpu_data()[i + 35], 0);
    }
  }
};  // class

TYPED_TEST_CASE(UnpoolingLayerTest, TestDtypesAndDevices);
 
TYPED_TEST(UnpoolingLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnpoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_kernel_size(3);
  unpooling_param->set_stride(2);
  UnpoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 13);
  EXPECT_EQ(this->blob_top_->width(), 11);
}

TYPED_TEST(UnpoolingLayerTest, TestSetupPadded) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnpoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_kernel_size(3);
  unpooling_param->set_stride(2);
  unpooling_param->set_pad(1);
  UnpoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 11);
  EXPECT_EQ(this->blob_top_->width(), 9);
}

TYPED_TEST(UnpoolingLayerTest, TestForward) {
    this->TestForwardSquare();
    this->TestForwardRectHigh();
}

// TYPED_TEST(UnpoolingLayerTest, PrintBackward) {
//   typedef typename TypeParam::Dtype Dtype;
//   LayerParameter layer_param;
//   UnpoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
//   unpooling_param->set_kernel_size(2);
//   unpooling_param->set_stride(1);
//   unpooling_param->set_pad(0);
//   this->blob_bottom_->Reshape(1, 1, 2, 4);
//   this->blob_bottom_->mutable_cpu_data()[0] = 9;
//   this->blob_bottom_->mutable_cpu_data()[1] = 5;
//   this->blob_bottom_->mutable_cpu_data()[2] = 5;
//   this->blob_bottom_->mutable_cpu_data()[3] = 8;
//   this->blob_bottom_->mutable_cpu_data()[4] = 9;
//   this->blob_bottom_->mutable_cpu_data()[5] = 5;
//   this->blob_bottom_->mutable_cpu_data()[6] = 5;
//   this->blob_bottom_->mutable_cpu_data()[7] = 8;
//   
//   this->blob_bottom_mask_->Reshape(1, 1, 2, 4);
//   this->blob_bottom_mask_->mutable_cpu_data()[0] = 5;
//   this->blob_bottom_mask_->mutable_cpu_data()[1] = 2;
//   this->blob_bottom_mask_->mutable_cpu_data()[2] = 2;
//   this->blob_bottom_mask_->mutable_cpu_data()[3] = 9;
//   this->blob_bottom_mask_->mutable_cpu_data()[4] = 5;
//   this->blob_bottom_mask_->mutable_cpu_data()[5] = 12;
//   this->blob_bottom_mask_->mutable_cpu_data()[6] = 12;
//   this->blob_bottom_mask_->mutable_cpu_data()[7] = 9;
//   
//   UnpoolingLayer<Dtype> layer(layer_param);
//   layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//   layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
//   
//   const int bottom_count = this->blob_bottom_->count();
//   const int top_count = this->blob_top_->count();
//   cout << "======================="  << endl;
//   const vector<int> top_shape = this->blob_top_->shape();
//   const vector<int> btm_shape = this->blob_bottom_->shape();
//   const vector<int> msk_shape = this->blob_bottom_mask_->shape();
//   
//   cout << "top shape: " << top_shape[0] << ", " << top_shape[1] << ", " << top_shape[2]<< ", " << top_shape[3] << endl;
//   cout << "btm shape: " << btm_shape[0] << ", " << btm_shape[1] << ", " << btm_shape[2]<< ", " << btm_shape[3] << endl;
//   cout << "msk shape: " << msk_shape[0] << ", " << msk_shape[1] << ", " << msk_shape[2]<< ", " << msk_shape[3] << endl;
//   cout << "======================="  << endl;
//   for (int i = 0; i < bottom_count; ++i) {
//     cout << "bottom data " << i << "/" << bottom_count-1 << " " << this->blob_bottom_->cpu_data()[i] << " " << this->blob_bottom_mask_->cpu_data()[i] << endl;
//   }
//   for (int i = 0; i < top_count; ++i) {
//     cout << "top data " << i << "/" << top_count-1 << " " << this->blob_top_->cpu_data()[i] << endl;
//   }
//   for (int i = 0; i < top_count; ++i) {
//     this->blob_top_->mutable_cpu_diff()[i] = i;
//   }
//   layer.Backward(this->blob_top_vec_, this->propagate_down_, this->blob_bottom_vec_);
//   for (int i = 0; i < bottom_count; ++i) {
//     cout << "bottom diff " << i << "/" << bottom_count-1 << " " << this->blob_bottom_->cpu_diff()[i] << " " << this->blob_bottom_mask_->cpu_diff()[i] << endl;
//   }
//   cout << "======================="  << endl;
// 
// }


TYPED_TEST(UnpoolingLayerTest, TestGradientSingle) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnpoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_kernel_size(2);
  unpooling_param->set_stride(1);
  unpooling_param->set_pad(0);
  //this->blob_bottom_->Reshape(1, 1, 2, 4);
  //this->blob_bottom_->mutable_cpu_data()[0] = 9;
  //this->blob_bottom_->mutable_cpu_data()[1] = 5;
  //this->blob_bottom_->mutable_cpu_data()[2] = 5;
  //this->blob_bottom_->mutable_cpu_data()[3] = 8;
  //this->blob_bottom_->mutable_cpu_data()[4] = 9;
  //this->blob_bottom_->mutable_cpu_data()[5] = 5;
  //this->blob_bottom_->mutable_cpu_data()[6] = 5;
  //this->blob_bottom_->mutable_cpu_data()[7] = 8;
  //
  //this->blob_bottom_mask_->Reshape(1, 1, 2, 4);
  //this->blob_bottom_mask_->mutable_cpu_data()[0] = 5;
  //this->blob_bottom_mask_->mutable_cpu_data()[1] = 2;
  //this->blob_bottom_mask_->mutable_cpu_data()[2] = 2;
  //this->blob_bottom_mask_->mutable_cpu_data()[3] = 9;
  //this->blob_bottom_mask_->mutable_cpu_data()[4] = 5;
  //this->blob_bottom_mask_->mutable_cpu_data()[5] = 12;
  //this->blob_bottom_mask_->mutable_cpu_data()[6] = 12;
  //this->blob_bottom_mask_->mutable_cpu_data()[7] = 9;
  
  UnpoolingLayer<Dtype> layer(layer_param);
  //layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-4, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
				  this->blob_top_vec_, 1);
}

// TYPED_TEST(UnpoolingLayerTest, TestGradientMax) {
//   typedef typename TypeParam::Dtype Dtype;
//   for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
//     for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
//       LayerParameter layer_param;
//       UnpoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
//       unpooling_param->set_kernel_h(kernel_h);
//       unpooling_param->set_kernel_w(kernel_w);
//       unpooling_param->set_stride(2);
//       unpooling_param->set_pad(1);
//       UnpoolingLayer<Dtype> layer(layer_param);
//       GradientChecker<Dtype> checker(1e-4, 1e-2);
//       checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//           this->blob_top_vec_);
//     }
//   }
// }

/** GPU TEST
 * ==========================================
 */

#ifdef USE_CUDNN
template <typename Dtype>
class GPUUnpoolingLayerTest : public GPUDeviceTest<Dtype> {
 protected:
  GPUUnpoolingLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_bottom_mask_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(2, 3, 6, 5);
    blob_bottom_mask_->Reshape(2, 3, 6, 5);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    // fill the values for the mask
    FillerParameter ufiller_param;
    ufiller_param.set_max(29);
    UniformFiller<Dtype> uniform_filler(ufiller_param);
    uniform_filler.Fill(this->blob_bottom_mask_);
    
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_mask_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~GPUUnpoolingLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
    delete blob_bottom_mask_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_mask_;  
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  
  // Test for 2x 2 square pooling layer
  void TestForwardSquare() {
    LayerParameter layer_param;
    UnpoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
    unpooling_param->set_kernel_size(2);
    const int num = 2;
    const int channels = 2;
    blob_bottom_->Reshape(num, channels, 3, 5);
    blob_bottom_mask_->Reshape(num, channels, 2, 4);
    // In (pooled map): 2x 2 channels of:
    //     [9 5 5 8]
    //     [9 5 5 8]
     for (int i = 0; i < 8 * num * channels; i += 8) {
      blob_bottom_->mutable_cpu_data()[i + 0] = 9;
      blob_bottom_->mutable_cpu_data()[i + 1] = 5;
      blob_bottom_->mutable_cpu_data()[i + 2] = 5;
      blob_bottom_->mutable_cpu_data()[i + 3] = 8;
      blob_bottom_->mutable_cpu_data()[i + 4] = 9;
      blob_bottom_->mutable_cpu_data()[i + 5] = 5;
      blob_bottom_->mutable_cpu_data()[i + 6] = 5;
      blob_bottom_->mutable_cpu_data()[i + 7] = 8;
    }
    
    // In (mask): 2x 2 channels of:
    //     [5  2  2 9]
    //     [5 12 12 9]
     for (int i = 0; i < 8 * num * channels; i += 8) {
      blob_bottom_mask_->mutable_cpu_data()[i + 0] =  5;
      blob_bottom_mask_->mutable_cpu_data()[i + 1] =  2;
      blob_bottom_mask_->mutable_cpu_data()[i + 2] =  2;
      blob_bottom_mask_->mutable_cpu_data()[i + 3] =  9;
      blob_bottom_mask_->mutable_cpu_data()[i + 4] =  5;
      blob_bottom_mask_->mutable_cpu_data()[i + 5] = 12;
      blob_bottom_mask_->mutable_cpu_data()[i + 6] = 12;
      blob_bottom_mask_->mutable_cpu_data()[i + 7] =  9;
    } 
    UnpoolingLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), 3);
    EXPECT_EQ(blob_top_->width(), 5);
    
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    
    // Expected output: 2x 2 channels of:
    //     [0 0 5 0 0]
    //     [9 0 0 0 8]
    //     [0 0 5 0 0]
    for (int i = 0; i < 15 * num * channels; i += 15) {
      EXPECT_EQ(blob_top_->cpu_data()[i +  0], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i +  1], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i +  2], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i +  3], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i +  4], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i +  5], 9);
      EXPECT_EQ(blob_top_->cpu_data()[i +  6], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i +  7], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i +  8], 0);      
      EXPECT_EQ(blob_top_->cpu_data()[i +  9], 8);      
      EXPECT_EQ(blob_top_->cpu_data()[i + 10], 0);      
      EXPECT_EQ(blob_top_->cpu_data()[i + 11], 0);      
      EXPECT_EQ(blob_top_->cpu_data()[i + 12], 5);      
      EXPECT_EQ(blob_top_->cpu_data()[i + 13], 0);      
      EXPECT_EQ(blob_top_->cpu_data()[i + 14], 0);      
    }  
  }
};

TYPED_TEST_CASE(GPUUnpoolingLayerTest, TestDtypes);

TYPED_TEST(GPUUnpoolingLayerTest, TestSetupGPU) {
  LayerParameter layer_param;
  UnpoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_kernel_size(3);
  unpooling_param->set_stride(2);
  UnpoolingLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 13);
  EXPECT_EQ(this->blob_top_->width(), 11);
}

TYPED_TEST(GPUUnpoolingLayerTest, TestSetupPaddedGPU) {
  LayerParameter layer_param;
  UnpoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_kernel_size(3);
  unpooling_param->set_stride(2);
  unpooling_param->set_pad(1);
  UnpoolingLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 11);
  EXPECT_EQ(this->blob_top_->width(), 9);
}

// TYPED_TEST(GPUUnpoolingLayerTest, TestForwardMaxGPU) {
//   this->TestForwardSquare();
// //   this->TestForwardRectHigh();
// //   this->TestForwardRectWide();
// }

// TYPED_TEST(GPUUnpoolingLayerTest, TestGradientMaxGPU) {
//   LayerParameter layer_param;
//   UnpoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
//   unpooling_param->set_kernel_size(2);
//   unpooling_param->set_stride(1);
//   unpooling_param->set_pad(0);
//   UnpoolingLayer<TypeParam> layer(layer_param);
//   GradientChecker<TypeParam> checker(1e-4, 1e-2);
//   checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//       this->blob_top_vec_);
// }

#endif
}  // namespace caffe
