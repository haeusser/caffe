#ifndef CAFFE_OPTIMIZATION_SPECTRAL_COMPONENTS_HPP_
#define CAFFE_OPTIMIZATION_SPECTRAL_COMPONENTS_HPP_

#include <boost/function.hpp>
#include <string>
#include <vector>

#include "caffe/blob.hpp"

namespace caffe {

/**
 * @brief Creates and applies spectral basis function transformations to blobs
 *
 * Must be created by a solver
 */
template <typename Dtype>
class SpectralComponentsManager {
 public:
  explicit SpectralComponentsManager(): temporary_blob_(new Blob<Dtype>) {}
  
  Blob<Dtype>* SpatialToSpectral(Caffe::Brew mode, const Blob<Dtype>* in_blob);
  Blob<Dtype>* SpectralToSpatial(Caffe::Brew mode, const Blob<Dtype>* in_blob);
  
  void SpatialToSpectral(Caffe::Brew mode, const Blob<Dtype>* in_blob, Blob<Dtype>* out_blob);
  void SpectralToSpatial(Caffe::Brew mode, const Blob<Dtype>* in_blob, Blob<Dtype>* out_blob);

 protected:
   
   enum transform_direction
   {
      SPATIAL_TO_SPECTRAL,
      SPECTRAL_TO_SPATIAL
   };

  Dtype real_dft2_get_value(int W, int w, int x, int H, int h, int y);
  Blob<Dtype>* transform(Caffe::Brew mode, transform_direction transf_dir, const Blob<Dtype>* in_blob, Blob<Dtype>* out_blob);
  
  Blob<Dtype> *getOrMakeBank(int W, int H);
  void fillBank(Blob<Dtype>* bank);

  // Maps width/height pair of kernel size to a basis function bank index
  map<pair<int,int>, Blob<Dtype>* > basis_functions_map_;
  
  Blob<Dtype>* temporary_blob_;

  DISABLE_COPY_AND_ASSIGN(SpectralComponentsManager);
};


}  // namespace caffe

#endif  // CAFFE_OPTIMIZATION_SPECTRAL_COMPONENTS_HPP_
