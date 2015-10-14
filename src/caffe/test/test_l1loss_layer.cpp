#include <cmath>
#include <cstdlib>
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
class L1LossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  L1LossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(5, 4, 3, 2)),
        blob_bottom_label_(new Blob<Dtype>(5, 4, 3, 2)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    filler.Fill(this->blob_bottom_label_);
    
    Dtype *bot0 = this->blob_bottom_data_->mutable_cpu_data();
    Dtype *bot1 = this->blob_bottom_label_->mutable_cpu_data();
    
    // If values are very close, we come to the 
    // non-differentiable part of L1, which is bad for gradient checking.
    // So lets avoid that:
    for(int i=0; i<this->blob_bottom_data_->count(); i++) {
        Dtype dist = fabs(bot0[i] - bot1[i]);
        if(dist < 2e-2) bot0[i] += 2e-2;
    }
        
    
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~L1LossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
  }
  
  void TestForward_simple() {
    // Get the loss without a specified objective weight -- should be
    // equivalent to explicitly specifiying a weight of 1.
    LayerParameter layer_param;
    L1LossLayer<Dtype> layer_weight_1(layer_param);
    layer_weight_1.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype loss_weight_1 =
        layer_weight_1.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // Get the loss again with a different objective weight; check that it is
    // scaled appropriately.
    const Dtype kLossWeight = 3.7;
    layer_param.add_loss_weight(kLossWeight);
    L1LossLayer<Dtype> layer_weight_2(layer_param);
    layer_weight_2.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype loss_weight_2 =
        layer_weight_2.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype kErrorMargin = 1e-5;
    EXPECT_NEAR(loss_weight_1 * kLossWeight, loss_weight_2, kErrorMargin);
    // Make sure the loss is non-trivial.
    const Dtype kNonTrivialAbsThresh = 1e-1;
    EXPECT_GE(fabs(loss_weight_1), kNonTrivialAbsThresh);
  }
  
  void TestForward_values() {
    LayerParameter layer_param;

    const Dtype kLossWeight = 0.00345;
    layer_param.add_loss_weight(kLossWeight);
    layer_param.mutable_l1_loss_param()->set_l2_per_location(false);
    
    L1LossLayer<Dtype> layer_weight_2(layer_param);
    layer_weight_2.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    
    const Dtype loss_weight_2 =
        layer_weight_2.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype kErrorMargin = 1e-5;
    
    //Compute reference
    Dtype refloss = 0;
    const Dtype *bot0 = blob_bottom_data_->cpu_data();
    const Dtype *bot1 = blob_bottom_label_->cpu_data();
    for(int c = 0; c < blob_bottom_data_->count(); c++) {
        refloss += fabs(bot0[c] - bot1[c]);
    }
    refloss /= (Dtype)blob_bottom_data_->num();
    
    EXPECT_NEAR(refloss * kLossWeight, loss_weight_2, kErrorMargin);
    
    // Make sure the loss is non-trivial.
    const Dtype kNonTrivialAbsThresh = 1e-1;
    EXPECT_GE(fabs(refloss), kNonTrivialAbsThresh);
  }
  
  void TestForward_values_l2() {
    LayerParameter layer_param;

    const Dtype kLossWeight = 5.67;
    layer_param.add_loss_weight(kLossWeight);
    layer_param.mutable_l1_loss_param()->set_l2_per_location(true);
    layer_param.mutable_l1_loss_param()->set_l2_prescale_by_channels(true);
    layer_param.mutable_l1_loss_param()->set_epsilon(1e-3);
    
    L1LossLayer<Dtype> layer_weight_2(layer_param);
    layer_weight_2.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    
    const Dtype loss_weight_2 =
        layer_weight_2.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype kErrorMargin = 1e-5;
    
    //Compute reference
    Dtype refloss = 0;
    int width = blob_bottom_data_->width();
    int height = blob_bottom_data_->height();
    int channels = blob_bottom_data_->channels();
    int num = blob_bottom_data_->num();
    
    const Dtype *bot0 = blob_bottom_data_->cpu_data();
    const Dtype *bot1 = blob_bottom_label_->cpu_data();
    for(int n = 0; n < num; n++) {
        for(int xy = 0; xy < width*height; xy++) {
            Dtype per_location_loss = 0;
            for(int c = 0; c < channels; c++) {
                int off = (n*channels + c)*height*width + xy;
                per_location_loss += (bot0[off]-bot1[off]) * (bot0[off]-bot1[off]);
            }
            refloss += sqrt(per_location_loss / (channels) + 1e-3);
        }
    }
    refloss /= (Dtype)num;
    
    EXPECT_NEAR(refloss * kLossWeight, loss_weight_2, kErrorMargin);
    
    // Make sure the loss is non-trivial.
    const Dtype kNonTrivialAbsThresh = 1e-1;
    EXPECT_GE(fabs(refloss), kNonTrivialAbsThresh);
  }
  
  void TestForward_values_l2_nans() {
    LayerParameter layer_param;

    const Dtype kLossWeight = 5.67;
    layer_param.add_loss_weight(kLossWeight);
    layer_param.mutable_l1_loss_param()->set_l2_per_location(true);
//     layer_param.mutable_l1_loss_param()->set_l2_prescale_by_channels(false); now false by default
    layer_param.mutable_l1_loss_param()->set_normalize_by_num_entries(true);
    
    L1LossLayer<Dtype> layer_weight_2(layer_param);
    
    Dtype* bot_data = this->blob_bottom_data_->mutable_cpu_data();
    Dtype tmp_val = bot_data[17];
    bot_data[17] = NAN;    
    
    layer_weight_2.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    
    const Dtype loss_weight_2 =
        layer_weight_2.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype kErrorMargin = 1e-5;
    
    //Compute reference
    Dtype refloss = 0;
    int num_entries = 0;
    int width = blob_bottom_data_->width();
    int height = blob_bottom_data_->height();
    int channels = blob_bottom_data_->channels();
    int num = blob_bottom_data_->num();
    
    const Dtype *bot0 = blob_bottom_data_->cpu_data();
    const Dtype *bot1 = blob_bottom_label_->cpu_data();
    for(int n = 0; n < num; n++) {
        for(int xy = 0; xy < width*height; xy++) {
            Dtype per_location_loss = 0;
            for(int c = 0; c < channels; c++) {
                int off = (n*channels + c)*height*width + xy;
                if (!std::isnan(bot0[off]-bot1[off])) {
                  num_entries++;
                  per_location_loss += (bot0[off]-bot1[off]) * (bot0[off]-bot1[off]);
                }                
            }
            refloss += sqrt(per_location_loss + 1e-2); // default epsilon is 1e-2
        }
    }
    refloss /= (Dtype)num_entries;
    refloss *= (Dtype)channels;
    
    EXPECT_NEAR(refloss * kLossWeight, loss_weight_2, kErrorMargin);
    
    // Make sure the loss is non-trivial.
    const Dtype kNonTrivialAbsThresh = 1e-1;
    EXPECT_GE(fabs(refloss), kNonTrivialAbsThresh);
    
    bot_data[17] = tmp_val;
  }
  
  void TestForward_values_singleblob() {
    LayerParameter layer_param;

    vector<Blob<Dtype>*> mybottom; // Only one blob
    mybottom.push_back(blob_bottom_data_);
    
    const Dtype kLossWeight = 0.02345;
    layer_param.add_loss_weight(kLossWeight);
    layer_param.mutable_l1_loss_param()->set_l2_per_location(false);
    
    L1LossLayer<Dtype> layer_weight_2(layer_param);
    layer_weight_2.SetUp(mybottom, this->blob_top_vec_);
    
    const Dtype loss_weight_2 =
        layer_weight_2.Forward(mybottom, this->blob_top_vec_);
    const Dtype kErrorMargin = 1e-5;
    
    //Compute reference
    Dtype refloss = 0;
    const Dtype *bot0 = blob_bottom_data_->cpu_data();
    for(int c = 0; c < blob_bottom_data_->count(); c++) {
        refloss += fabs(bot0[c]);
    }
    refloss /= (Dtype)blob_bottom_data_->num();
    
    EXPECT_NEAR(refloss * kLossWeight, loss_weight_2, kErrorMargin);
    
    // Make sure the loss is non-trivial.
    const Dtype kNonTrivialAbsThresh = 1e-1;
    EXPECT_GE(fabs(refloss), kNonTrivialAbsThresh);
  }
  
  void TestForward_l2_per_location() {
    // Get the loss without a specified objective weight -- should be
    // equivalent to explicitly specifiying a weight of 1.
    LayerParameter layer_param;
    layer_param.mutable_l1_loss_param()->set_l2_per_location(true);
    L1LossLayer<Dtype> layer_weight_1(layer_param);
    layer_weight_1.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype loss_weight_1 =
        layer_weight_1.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // Get the loss again with a different objective weight; check that it is
    // scaled appropriately.
    const Dtype kLossWeight = 3.7;
    layer_param.add_loss_weight(kLossWeight);
    L1LossLayer<Dtype> layer_weight_2(layer_param);
    layer_weight_2.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype loss_weight_2 =
        layer_weight_2.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype kErrorMargin = 1e-5;
    EXPECT_NEAR(loss_weight_1 * kLossWeight, loss_weight_2, kErrorMargin);
    // Make sure the loss is non-trivial.
    const Dtype kNonTrivialAbsThresh = 1e-1;
    EXPECT_GE(fabs(loss_weight_1), kNonTrivialAbsThresh);
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(L1LossLayerTest, TestDtypesAndDevices);

TYPED_TEST(L1LossLayerTest, TestForward_simple) {
  if(Caffe::mode()==Caffe::CPU)
  {
      LOG(INFO) << "Skipping CPU test";
      return;
  }
  this->TestForward_simple();
}

TYPED_TEST(L1LossLayerTest, TestForward_values) {
  if(Caffe::mode()==Caffe::CPU)
  {
      LOG(INFO) << "Skipping CPU test";
      return;
  }
  this->TestForward_values();
  this->TestForward_values_singleblob();
}

TYPED_TEST(L1LossLayerTest, TestForward_values_l2) {
  if(Caffe::mode()==Caffe::CPU)
  {
      LOG(INFO) << "Skipping CPU test";
      return;
  }
  this->TestForward_values_l2();
}

TYPED_TEST(L1LossLayerTest, TestForward_l2_per_location) {
  if(Caffe::mode()==Caffe::CPU)
  {
      LOG(INFO) << "Skipping CPU test";
      return;
  }
  this->TestForward_l2_per_location();
}

TYPED_TEST(L1LossLayerTest, TestGradient_simple) {
  if(Caffe::mode()==Caffe::CPU)
  {
      LOG(INFO) << "Skipping CPU test";
      return;
  }
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_l1_loss_param()->set_l2_per_location(false);
  const Dtype kLossWeight = 3.7;
  layer_param.add_loss_weight(kLossWeight);
  L1LossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(L1LossLayerTest, TestGradient_l2_per_location) {
  if(Caffe::mode()==Caffe::CPU)
  {
      LOG(INFO) << "Skipping CPU test";
      return;
  }
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_l1_loss_param()->set_l2_per_location(true);
  const Dtype kLossWeight = 3.7;
  layer_param.add_loss_weight(kLossWeight);
  L1LossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(L1LossLayerTest, TestForward_values_l2_nans) {
  if(Caffe::mode()==Caffe::CPU)
  {
      LOG(INFO) << "Skipping CPU test";
      return;
  }
  this->TestForward_values_l2_nans();
}

}  // namespace caffe
