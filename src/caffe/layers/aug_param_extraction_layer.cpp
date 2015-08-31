// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <vector>
#include <cmath>

#include "google/protobuf/descriptor.h"
#include "google/protobuf/descriptor.h"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/math_functions.hpp"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

#include <iostream>
#include <fstream>
#include <omp.h>

using std::max;

namespace caffe {
  
template <typename Dtype>
void AugParamExtractionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
}

template <typename Dtype>
void AugParamExtractionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
  AugParamExtractionParameter layer_param = this->layer_param_.aug_param_extraction_param();
  
  CHECK(layer_param.has_extract_param()) << "AugParamExtractionLayer layer: Must specify extract_param";
  CHECK_EQ(bottom.size(), 1) << "AugParamExtractionLayer layer takes augparams as input blob";
  CHECK_EQ(top.size(), 1) << "AugParamExtractionLayer layer outputs one blob";

  extract_param_ = layer_param.extract_param();
  multiplier_ = layer_param.multiplier();
  
  /*if(extract_param_ != AugParamExtractionParameter_ExtractParam_MIRROR) {
    LOG(FATAL) << "For now only MIRROR parameter is supported.";
  }*/
      
  const int num = bottom[0]->num();
//   const int channels = bottom[0]->channels();

  if(extract_param_ == AugParamExtractionParameter_ExtractParam_MIRROR) {
    (top)[0]->Reshape(num,2,1,1); // For mirror 2 channels
  } else if(extract_param_ == AugParamExtractionParameter_ExtractParam_ROTATION) {
    (top)[0]->Reshape(num,2,1,1); // For rotation 2 channels (positive and negative rotation separated)
  } else if(extract_param_ == AugParamExtractionParameter_ExtractParam_ROTATION_SCALAR) {
    (top)[0]->Reshape(num,1,1,1); // For rotation 1 channels scalar
  } else if(extract_param_ == AugParamExtractionParameter_ExtractParam_ROTATION_COSSIN) {
    (top)[0]->Reshape(num,2,1,1); // For cos sin rotation 2 channels
  } else if(extract_param_ == AugParamExtractionParameter_ExtractParam_ROTATION_90DEG_BINS) {
    (top)[0]->Reshape(num,4,1,1); // For cos sin rotation 2 channels
  } else if(extract_param_ == AugParamExtractionParameter_ExtractParam_ROTATION_90DEG_CLASS) {
    (top)[0]->Reshape(num,1,1,1); //class in [0,1,2,3]
  } else if(extract_param_ == AugParamExtractionParameter_ExtractParam_ROTATION_45DEG_REL) {
    (top)[0]->Reshape(num,2,1,1); // +-45deg rotation relative to nearest rounded 90deg (2 channels as for ROTATION)
  }
  
  // Set up coeff blobs
  all_coeffs1_.ReshapeLike(*bottom[0]);

  // How many params exist in general?
  AugmentationCoeff coeff;
  num_params_ = coeff.GetDescriptor()->field_count();

  // = Coeff transformation matrix cache for one batch
  coeff_matrices1_.reset(new SyncedMemory(num * sizeof(typename AugmentationLayerBase<Dtype>::tTransMat)));

  
}

template <typename Dtype>
void AugParamExtractionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{  
  int num = (bottom)[0]->num();
  Dtype* top_data = (top)[0]->mutable_cpu_data();
  
  all_coeffs1_.ShareData(*bottom[0]);
  const Dtype* my_params1 = all_coeffs1_.cpu_data();

  for (int item_id = 0; item_id < num; ++item_id) {
    AugmentationCoeff coeff; 

    // Load the previously generated coeffs (either they are from another layer or generated above)
    AugmentationLayerBase<Dtype>::array_to_coeff(my_params1 + item_id * num_params_, coeff);
    
    if(extract_param_ == AugParamExtractionParameter_ExtractParam_MIRROR) {
      if(coeff.mirror()) {
        top_data[item_id * 2 + 0] = multiplier_;
        top_data[item_id * 2 + 1] = 0.0;
      } else {
        top_data[item_id * 2 + 0] = 0.0;
        top_data[item_id * 2 + 1] = multiplier_;
      }
    } else if(extract_param_ == AugParamExtractionParameter_ExtractParam_ROTATION) {
      float angle = coeff.angle();
      if(angle >= 0) {
        top_data[item_id * 2 + 0] = angle * multiplier_;
        top_data[item_id * 2 + 1] = 0.0;
      } else {
        top_data[item_id * 2 + 0] = 0.0;
        top_data[item_id * 2 + 1] = -angle * multiplier_;
      }
    } else if(extract_param_ == AugParamExtractionParameter_ExtractParam_ROTATION_SCALAR) {
      float angle = coeff.angle();
      top_data[item_id * 1 + 0] = angle * multiplier_;
    } else if(extract_param_ == AugParamExtractionParameter_ExtractParam_ROTATION_COSSIN) {
      float angle = coeff.angle();
      float anglecos = cos(angle);
      float anglesin = sin(angle);
      top_data[item_id * 2 + 0] = anglecos;
      top_data[item_id * 2 + 1] = anglesin;
    } else if(extract_param_ == AugParamExtractionParameter_ExtractParam_ROTATION_90DEG_BINS) {
      int introt = ((int)(round(coeff.angle() / 1.570796326)) + 8) % 4; // In pi/2
      top_data[item_id * 4 + 0] = (introt == 0) ? 1.0 : 0.0;
      top_data[item_id * 4 + 1] = (introt == 1) ? 1.0 : 0.0;
      top_data[item_id * 4 + 2] = (introt == 2) ? 1.0 : 0.0;
      top_data[item_id * 4 + 3] = (introt == 3) ? 1.0 : 0.0;
      
    } else if(extract_param_ == AugParamExtractionParameter_ExtractParam_ROTATION_90DEG_CLASS) {
      int introt = ((int)(round(coeff.angle() / 1.570796326)) + 8) % 4; // In pi/2
      top_data[item_id * 1 + 0] = introt;
    } else if(extract_param_ == AugParamExtractionParameter_ExtractParam_ROTATION_45DEG_REL) {
      float baserot = round(coeff.angle() / 1.570796326) * 1.570796326;
      float angle = coeff.angle() - baserot;
      if(angle >= 0) {
        top_data[item_id * 2 + 0] = angle * multiplier_;
        top_data[item_id * 2 + 1] = 0.0;
      } else {
        top_data[item_id * 2 + 0] = 0.0;
        top_data[item_id * 2 + 1] = -angle * multiplier_;
      }
    }

    
  }
  
}

template <typename Dtype>
void AugParamExtractionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  Forward_cpu(bottom, top);
}

INSTANTIATE_CLASS(AugParamExtractionLayer);


}  // namespace caffe
