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
class LpqLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:

  /**
   * --MEMO--
   * Blob(num, channels, height, width)
   */
   
  /**
   * Constructor
   */
  LpqLossLayerTest()
    : blob_bottom_data_(new Blob<Dtype>(5,4,3,2)),
      blob_bottom_label_(new Blob<Dtype>(5,4,3,2)),
      blob_top_loss_(new Blob<Dtype>()) 
  {
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
      Dtype dist = std::abs(bot0[i] - bot1[i]);
      if(dist < 2e-2) bot0[i] += 4e-2;
    }
    
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  
  /**
   * Destructor
   */
  virtual ~LpqLossLayerTest() 
  {
    if (blob_bottom_data_)  delete blob_bottom_data_;
    if (blob_bottom_label_) delete blob_bottom_label_;
    if (blob_top_loss_)     delete blob_top_loss_;
  }
  
  /**
   * Test layer setup
   */
  void TestForward_simple() 
  {
    // Get the loss without a specified objective weight -- should be
    // equivalent to explicitly specifiying a weight of 1.
    LayerParameter layer_param;
    
    //layer_param.mutable_l1_loss_param()->set_l2_per_location(true);
    
    layer_param.mutable_lpq_loss_param()->add_pq_episode_starts_at_iter(0);
    layer_param.mutable_lpq_loss_param()->add_p((Dtype)1);
    layer_param.mutable_lpq_loss_param()->add_q((Dtype)1);
    
    LpqLossLayer<Dtype> layer_weight_1(layer_param);
    layer_weight_1.SetUp(this->blob_bottom_vec_, 
                         this->blob_top_vec_);
    const Dtype loss_weight_1 = layer_weight_1.Forward(this->blob_bottom_vec_, 
                                                       this->blob_top_vec_);

    // Get the loss again with a different objective weight; check that it is
    // scaled appropriately.
    const Dtype kLossWeight = (Dtype)3.7;
    layer_param.add_loss_weight(kLossWeight);
    LpqLossLayer<Dtype> layer_weight_2(layer_param);
    layer_weight_2.SetUp(this->blob_bottom_vec_, 
                         this->blob_top_vec_);
    const Dtype loss_weight_2 = layer_weight_2.Forward(this->blob_bottom_vec_, 
                                                       this->blob_top_vec_);
    const Dtype kErrorMargin = (Dtype)1e-5;
    EXPECT_NEAR(loss_weight_1 * kLossWeight, loss_weight_2, kErrorMargin);
    // Make sure the loss is non-trivial.
    const Dtype kNonTrivialAbsThresh = (Dtype)1e-1;
    EXPECT_GE(fabs(loss_weight_1), kNonTrivialAbsThresh);
  }
  
  /**
   * Test loss (forward pass): L1
   */
  void TestForward_values_l1(Dtype q_eps) 
  {
    LayerParameter layer_param;
    
    layer_param.mutable_lpq_loss_param()->set_p_epsilon(Dtype(0));
    layer_param.mutable_lpq_loss_param()->set_q_epsilon(q_eps);
    layer_param.mutable_lpq_loss_param()->add_p((Dtype)1);
    layer_param.mutable_lpq_loss_param()->add_q((Dtype)1);

    const Dtype kLossWeight = (Dtype)0.00345;
    layer_param.add_loss_weight(kLossWeight);
    
    LpqLossLayer<Dtype> layer_weight_2(layer_param);
    layer_weight_2.SetUp(this->blob_bottom_vec_, 
                         this->blob_top_vec_);
    const Dtype loss_weight_2 = layer_weight_2.Forward(this->blob_bottom_vec_, 
                                                       this->blob_top_vec_);
    const Dtype kErrorMargin = (Dtype)1e-5;
    
    /// Compute reference loss
    Dtype refloss = (Dtype)0;
    const Dtype *bot0 = blob_bottom_data_->cpu_data();
    const Dtype *bot1 = blob_bottom_label_->cpu_data();
    //blob_bottom_data_->print("DATA");
    //blob_bottom_label_->print("LABEL");
    //blob_top_loss_->print("LOSS");
    for(int c = 0; c < blob_bottom_data_->count(); ++c) {
      refloss += std::abs(bot0[c] - bot1[c]) + 
                 q_eps / blob_bottom_data_->channels();
    }
    refloss /= (Dtype)blob_bottom_data_->num();
    
    EXPECT_NEAR(refloss * kLossWeight, loss_weight_2, kErrorMargin);
    
    /// Make sure the loss is non-trivial.
    const Dtype kNonTrivialAbsThresh = 1e-1;
    EXPECT_GE(fabs(refloss), kNonTrivialAbsThresh);
  }
  
  /**
   * Test loss (forward pass): L2_2
   */
  void TestForward_values_l2(Dtype q_eps) 
  {
    LayerParameter layer_param;
    
    layer_param.mutable_lpq_loss_param()->set_p_epsilon(Dtype(0));
    layer_param.mutable_lpq_loss_param()->set_q_epsilon(q_eps);
    layer_param.mutable_lpq_loss_param()->add_p((Dtype)2);
    layer_param.mutable_lpq_loss_param()->add_q((Dtype)(1./2.));

    const Dtype kLossWeight = (Dtype)1;
    layer_param.add_loss_weight(kLossWeight);
    
    LpqLossLayer<Dtype> layer_weight_2(layer_param);
    layer_weight_2.SetUp(this->blob_bottom_vec_, 
                         this->blob_top_vec_);
    const Dtype loss_weight_2 = layer_weight_2.Forward(this->blob_bottom_vec_, 
                                                       this->blob_top_vec_);
    const Dtype kErrorMargin = (Dtype)1e-5;
    
    int width = blob_bottom_data_->width();
    int height = blob_bottom_data_->height();
    int channels = blob_bottom_data_->channels();
    int num = blob_bottom_data_->num();
    const Dtype *bot0 = blob_bottom_data_->cpu_data();
    const Dtype *bot1 = blob_bottom_label_->cpu_data();

    /// Compute reference loss
    Dtype refloss = (Dtype)0;
    for(int n = 0; n < num; ++n) {
      for(int xy = 0; xy < width*height; ++xy) {
        Dtype per_location_loss = 0;
        for(int c = 0; c < channels; ++c) {
          int off = (n*channels + c)*height*width + xy;
          per_location_loss += (bot0[off]-bot1[off]) * (bot0[off]-bot1[off]);
        }
        float summed_scaled  = per_location_loss;
        summed_scaled += q_eps;
        //if(summed_scaled < plateau*plateau) summed_scaled = 0;
        refloss += sqrt(summed_scaled);
      }
    }
    refloss /= (Dtype)num;
    
    EXPECT_NEAR(refloss * kLossWeight, loss_weight_2, kErrorMargin);
    
    /// Make sure the loss is non-trivial.
    const Dtype kNonTrivialAbsThresh = 1e-1;
    EXPECT_GE(fabs(refloss), kNonTrivialAbsThresh);
  }
  
  /**
   * Test loss (forward pass): L2_2 with normalize_by_num_entries
   */
  void TestForward_values_l2_normByNumEntries(Dtype q_eps)
  {
    LayerParameter layer_param;
    
    layer_param.mutable_l1_loss_param()->set_normalize_by_num_entries(true);
    
    layer_param.mutable_lpq_loss_param()->set_p_epsilon(Dtype(0));
    layer_param.mutable_lpq_loss_param()->set_q_epsilon(q_eps);
    layer_param.mutable_lpq_loss_param()->add_p((Dtype)2);
    layer_param.mutable_lpq_loss_param()->add_q((Dtype)(1./2.));

    const Dtype kLossWeight = (Dtype)1;
    layer_param.add_loss_weight(kLossWeight);
    
    LpqLossLayer<Dtype> layer_weight_2(layer_param);
    layer_weight_2.SetUp(this->blob_bottom_vec_, 
                         this->blob_top_vec_);
    const Dtype loss_weight_2 = layer_weight_2.Forward(this->blob_bottom_vec_, 
                                                       this->blob_top_vec_);
    const Dtype kErrorMargin = (Dtype)1e-5;
    
    /**
     * --MEMO--
     * Blob(num, channels, height, width)
     */
    
    int width = blob_bottom_data_->width();
    int height = blob_bottom_data_->height();
    int channels = blob_bottom_data_->channels();
    int num = blob_bottom_data_->num();
    const Dtype *bot0 = blob_bottom_data_->cpu_data();
    const Dtype *bot1 = blob_bottom_label_->cpu_data();

    /// Compute reference loss
    Dtype refloss = (Dtype)0;
    for(int n = 0; n < num; ++n) {
      for(int xy = 0; xy < width*height; ++xy) {
        Dtype per_location_loss = 0;
        for(int c = 0; c < channels; ++c) {
          int off = (n*channels + c)*height*width + xy;
          per_location_loss += (bot0[off]-bot1[off]) * (bot0[off]-bot1[off]);
        }
        float summed_scaled  = per_location_loss;
        summed_scaled += q_eps;
        //if(summed_scaled < plateau*plateau) summed_scaled = 0;
        refloss += sqrt(summed_scaled);
      }
    }
    refloss /= (Dtype)(width*height*num);
    
    EXPECT_NEAR(refloss * kLossWeight, loss_weight_2, kErrorMargin);
    
    /// Make sure the loss is non-trivial.
    const Dtype kNonTrivialAbsThresh = 1e-1;
    EXPECT_GE(fabs(refloss), kNonTrivialAbsThresh);
  }
  
//   void TestForward_values_l2(float plateau) {
//     LayerParameter layer_param;
// 
//     const Dtype kLossWeight = 5.67;
//     layer_param.add_loss_weight(kLossWeight);
//     layer_param.mutable_l1_loss_param()->set_l2_per_location(true);
//     layer_param.mutable_l1_loss_param()->set_l2_prescale_by_channels(true);
//     layer_param.mutable_l1_loss_param()->set_epsilon(1e-3);
//     layer_param.mutable_l1_loss_param()->set_plateau(plateau);
//     
//     LpqLossLayer<Dtype> layer_weight_2(layer_param);
//     layer_weight_2.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//     
//     const Dtype loss_weight_2 =
//         layer_weight_2.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
//     const Dtype kErrorMargin = 1e-5;
//     
//     //Compute reference
//     Dtype refloss = 0;
//     int width = blob_bottom_data_->width();
//     int height = blob_bottom_data_->height();
//     int channels = blob_bottom_data_->channels();
//     int num = blob_bottom_data_->num();
//     
//     const Dtype *bot0 = blob_bottom_data_->cpu_data();
//     const Dtype *bot1 = blob_bottom_label_->cpu_data();
//     for(int n = 0; n < num; n++) {
//         for(int xy = 0; xy < width*height; xy++) {
//             Dtype per_location_loss = 0;
//             for(int c = 0; c < channels; c++) {
//                 int off = (n*channels + c)*height*width + xy;
//                 per_location_loss += (bot0[off]-bot1[off]) * (bot0[off]-bot1[off]);
//             }
//             float summed_scaled  = per_location_loss / channels;
//             
//             if(summed_scaled < plateau*plateau) summed_scaled = 0;
//             
//             refloss += sqrt(summed_scaled + 1e-3);
//         }
//     }
//     refloss /= (Dtype)num;
//     
//     EXPECT_NEAR(refloss * kLossWeight, loss_weight_2, kErrorMargin);
//     
//     // Make sure the loss is non-trivial.
//     const Dtype kNonTrivialAbsThresh = 1e-1;
//     EXPECT_GE(fabs(refloss), kNonTrivialAbsThresh);
//   }
  
  void TestForward_values_l1_nans() {
    LayerParameter layer_param;

    const Dtype kLossWeight = 5.67;
    layer_param.add_loss_weight(kLossWeight);
    layer_param.mutable_l1_loss_param()->set_normalize_by_num_entries(true);
    
    layer_param.mutable_lpq_loss_param()->set_p_epsilon((Dtype)0);
    layer_param.mutable_lpq_loss_param()->set_q_epsilon((Dtype)1e-2);
    layer_param.mutable_lpq_loss_param()->add_p((Dtype)2);
    layer_param.mutable_lpq_loss_param()->add_q((Dtype)(1./2.));
    
    LpqLossLayer<Dtype> layer_weight_2(layer_param);
    
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
    
    layer_param.mutable_lpq_loss_param()->set_p_epsilon((Dtype)0);
    layer_param.mutable_lpq_loss_param()->set_q_epsilon((Dtype)0);
    layer_param.mutable_lpq_loss_param()->add_p((Dtype)1);
    layer_param.mutable_lpq_loss_param()->add_q((Dtype)1);
    
    LpqLossLayer<Dtype> layer_weight_2(layer_param);
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

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(LpqLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(LpqLossLayerTest, TestForward_simple) {
  if(Caffe::mode()==Caffe::CPU)
  {
      LOG(INFO) << "Skipping CPU test";
      return;
  }
  this->TestForward_simple();
}

TYPED_TEST(LpqLossLayerTest, TestForward_values_l1) {
  if(Caffe::mode()==Caffe::CPU)
  {
      LOG(INFO) << "Skipping CPU test";
      return;
  }
  this->TestForward_values_l1(0);
  this->TestForward_values_l1(1e-1);
}

TYPED_TEST(LpqLossLayerTest, TestForward_values_l2) {
  if(Caffe::mode()==Caffe::CPU)
  {
      LOG(INFO) << "Skipping CPU test";
      return;
  }
  this->TestForward_values_l2(0);
  this->TestForward_values_l2(1e-1);
}

TYPED_TEST(LpqLossLayerTest, TestForward_values_l2_normByNumEntries) {
  if(Caffe::mode()==Caffe::CPU)
  {
      LOG(INFO) << "Skipping CPU test";
      return;
  }
  this->TestForward_values_l2_normByNumEntries(0);
  this->TestForward_values_l2_normByNumEntries(1e-1);
}

TYPED_TEST(LpqLossLayerTest, TestForward_values_l1_nans) {
  if(Caffe::mode()==Caffe::CPU)
  {
      LOG(INFO) << "Skipping CPU test";
      return;
  }
  this->TestForward_values_l1_nans();
}

TYPED_TEST(LpqLossLayerTest, TestForward_values_singleblob) {
  if(Caffe::mode()==Caffe::CPU)
  {
      LOG(INFO) << "Skipping CPU test";
      return;
  }
  this->TestForward_values_singleblob();
}

TYPED_TEST(LpqLossLayerTest, TestGradient_L1) {
  if(Caffe::mode()==Caffe::CPU)
  {
      LOG(INFO) << "Skipping CPU test";
      return;
  }
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_lpq_loss_param()->set_p_epsilon(Dtype(0));
  layer_param.mutable_lpq_loss_param()->set_q_epsilon(Dtype(0));
  layer_param.mutable_lpq_loss_param()->add_p((Dtype)1);
  layer_param.mutable_lpq_loss_param()->add_q((Dtype)1);
  const Dtype kLossWeight = 3.7;
  layer_param.add_loss_weight(kLossWeight);
  LpqLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  
  GradientChecker<Dtype> checker(1e-2, 1e-3, 1700);
  checker.CheckGradientExhaustive(&layer, 
                                  this->blob_bottom_vec_,
                                  this->blob_top_vec_);
}

TYPED_TEST(LpqLossLayerTest, TestGradient_L2_1) {
  if(Caffe::mode()==Caffe::CPU)
  {
      LOG(INFO) << "Skipping CPU test";
      return;
  }
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_lpq_loss_param()->set_p_epsilon(Dtype(0));
  layer_param.mutable_lpq_loss_param()->set_q_epsilon(Dtype(0));
  layer_param.mutable_lpq_loss_param()->add_p((Dtype)2);
  layer_param.mutable_lpq_loss_param()->add_q((Dtype)1);
  const Dtype kLossWeight = 3.7;
  layer_param.add_loss_weight(kLossWeight);
  LpqLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  
  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701);
  checker.CheckGradientExhaustive(&layer, 
                                  this->blob_bottom_vec_,
                                  this->blob_top_vec_);
}

TYPED_TEST(LpqLossLayerTest, TestGradient_L2_2) {
  if(Caffe::mode()==Caffe::CPU)
  {
      LOG(INFO) << "Skipping CPU test";
      return;
  }
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_lpq_loss_param()->set_p_epsilon(Dtype(0));
  layer_param.mutable_lpq_loss_param()->set_q_epsilon(Dtype(0));
  layer_param.mutable_lpq_loss_param()->add_p((Dtype)2);
  layer_param.mutable_lpq_loss_param()->add_q((Dtype)(1./2.));
  const Dtype kLossWeight = 3.7;
  layer_param.add_loss_weight(kLossWeight);
  LpqLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  
//   LOG(INFO) << "TEST BODY";
//   this->blob_bottom_data_->print("DATA");
//   this->blob_bottom_label_->print("LABEL");
//   this->blob_top_loss_->print("LOSS");

  GradientChecker<Dtype> checker(1e-2, 1e-3, 1702);
  checker.CheckGradientExhaustive(&layer, 
                                  this->blob_bottom_vec_,
                                  this->blob_top_vec_);
}

// TYPED_TEST(LpqLossLayerTest, TestForward_values_l2_plateau) {
//   if(Caffe::mode()==Caffe::CPU)
//   {
//       LOG(INFO) << "Skipping CPU test";
//       return;
//   }
//   this->TestForward_values_l2(1.0);
// }
// 
// TYPED_TEST(LpqLossLayerTest, TestGradient_l2_per_location_plateau) {
//   if(Caffe::mode()==Caffe::CPU)
//   {
//       LOG(INFO) << "Skipping CPU test";
//       return;
//   }
//   typedef typename TypeParam::Dtype Dtype;
//   LayerParameter layer_param;
//   layer_param.mutable_l1_loss_param()->set_l2_per_location(true);
//   layer_param.mutable_l1_loss_param()->set_plateau(1.0);
//   const Dtype kLossWeight = 3.7;
//   layer_param.add_loss_weight(kLossWeight);
//   LpqLossLayer<Dtype> layer(layer_param);
//   layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//   GradientChecker<Dtype> checker(1e-2, 1e-3, 1701);
//   checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//       this->blob_top_vec_);
// }
// 
// TYPED_TEST(LpqLossLayerTest, TestForward_values_l2_nans) {
//   if(Caffe::mode()==Caffe::CPU)
//   {
//       LOG(INFO) << "Skipping CPU test";
//       return;
//   }
//   this->TestForward_values_l2_nans();
// }

}  // namespace caffe
