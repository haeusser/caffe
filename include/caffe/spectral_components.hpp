#ifndef CAFFE_OPTIMIZATION_SPECTRAL_COMPONENTS_HPP_
#define CAFFE_OPTIMIZATION_SPECTRAL_COMPONENTS_HPP_

#include <boost/function.hpp>
#include <string>
#include <vector>

#include "caffe/net.hpp"

namespace caffe {

template <typename Dtype> class Solver;

/**
 * @brief Creates and applies spectral basis function transformations to blobs
 *
 * Must be created by a solver
 */
template <typename Dtype>
class SpectralComponentsManager {
 public:
  explicit SpectralComponentsManager(const Solver<Dtype>* solver);
  
  void SpatialToSpectral(Blob<Dtype>*, Brew mode);
  void SpectralToSpatial(Blob<Dtype>*, Brew mode);

 protected:

  const Solver<Dtype>* const solver_;
  
  float real_dft2(int W, int w, int x, int H, int h, int y);
  
  Blob<Dtype> *getOrMakeBank(int w, int h);

  // Maps width/height pair of kernel size to a basis function bank index
  map<pair<int,int>, Blob<Dtype>* > basis_functions_map_;
  
  map<pair<int,int>, Blob<Dtype>* > temporary_blobs_;

  DISABLE_COPY_AND_ASSIGN(SpectralComponentsManager);
};


}  // namespace caffe

#endif  // CAFFE_OPTIMIZATION_SPECTRAL_COMPONENTS_HPP_
