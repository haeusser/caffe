#include <cstdio>

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/spectral_components.hpp"

namespace caffe {

// Compute one pixel of a filter bank of W*H filters with size W*H each
// (W,H) = width and height of filter
// (w,h) = position in filter bank ([0,W-1],[0,H-1])
// (x,y) = pixel position in the filter
template <typename Dtype>
Dtype SpectralComponentsManager<Dtype>::real_dft2_get_value(int W, int w, int x, int H, int h, int y) {
  Dtype fx = (Dtype)(x);
  Dtype fy = (Dtype)(y);
  Dtype fw = (Dtype)(w);
  Dtype fh = (Dtype)(h);
  Dtype fH = (Dtype)(H);
  Dtype fW = (Dtype)(W);
  
  Dtype pi = 3.141592653589793238462643383279;
  
  if ((2*h <= H && 2*w <= W) || (2*h > H && (w > 0 && 2*w < W)))
    return cos(2*pi*(fw*fx/fW + fh*fy/fH));
  if (2*h > H && (w == 0 || 2*w == W))
    return sin(2*pi*(fw*fx/fW + (fh-(Dtype)(H/2))*fy/fH));
  if (2*w > W)
    return sin(2*pi*((fw-(Dtype)(W/2))*fx/fW + fh*fy/fH));
  
  return 0.;
}


// ----------------------------------------------------------------------------------------

template <typename Dtype>
Blob<Dtype> *SpectralComponentsManager<Dtype>::getOrMakeBank(int W, int H) {
  
  Blob<Dtype> *bank_blob = basis_functions_map_[make_pair(W,H)]; // get or create
  
  if(bank_blob == NULL) {
    bank_blob = new Blob<Dtype>();
    bank_blob->Reshape(1,W*H,W,H);
    fillBank(bank_blob);
    basis_functions_map_[make_pair(W,H)] = bank_blob;
  }
  return bank_blob;
}

template <typename Dtype>
const Dtype *SpectralComponentsManager<Dtype>::getBlobPart(const Blob<Dtype> *blob, Caffe::Brew mode, blob_part part) {
  if (mode == Caffe::CPU) {
    if (part == BLOB_DATA) return blob->cpu_data();
    else if (part == BLOB_DIFF) return blob->cpu_diff();
  } else if (mode == Caffe::GPU) {
    if (part == BLOB_DATA) return blob->gpu_data();
    else if (part == BLOB_DIFF) return blob->gpu_diff();
  } else LOG(FATAL) << "Unknown mode " << mode;
  return NULL;
}

template <typename Dtype>
Dtype *SpectralComponentsManager<Dtype>::getMutableBlobPart(Blob<Dtype> *blob, Caffe::Brew mode, blob_part part) {
  if (mode == Caffe::CPU) {
    if (part == BLOB_DATA) return blob->mutable_cpu_data();
    else if (part == BLOB_DIFF) return blob->mutable_cpu_diff();
  } else if (mode == Caffe::GPU) {
    if (part == BLOB_DATA) return blob->mutable_gpu_data();
    else if (part == BLOB_DIFF) return blob->mutable_gpu_diff();
  } else LOG(FATAL) << "Unknown mode " << mode;
  return NULL;
}

template <typename Dtype>
void SpectralComponentsManager<Dtype>::fillBank(Blob<Dtype>* bank) {
  int W = bank->shape()[3];
  int H = bank->shape()[2];
  CHECK_EQ(bank->shape()[1], W*H) << "Fourier spatial bank should have W*H=" << W*H << " filters, not " << bank->shape()[1];
  CHECK_EQ(bank->shape()[0], 1);
  Dtype *data = bank->mutable_cpu_data();
  
  // for all filters in the bank
  for (int w=0; w<W; w++)
    for (int h=0; h<H; h++) {
      // index where he current filter starts
      int start_ind = (h * W + w) * H * W; 
      // fill in the values
      for (int y=0; y<H; y++) {
        for (int x=0; x<W; x++) {
          int ind = start_ind + y * W + x;
          data[ind] = this->real_dft2_get_value(W, w, x, H, h, y);        
        }        
      }
      // normalize the filter to unit Euclidean norm
      float norm_coeff = caffe::caffe_cpu_dot(W*H, data + start_ind, data + start_ind);
      norm_coeff = sqrt(norm_coeff);
      caffe_cpu_scale<float>(W*H, 1./norm_coeff, data + start_ind, data + start_ind);
    }
}

// does the transform both ways, depending on transform_direction being SPATIAL_TO_SPECTRAL or SPECTRAL_TO_SPATIAL
template <typename Dtype>
Blob<Dtype>* SpectralComponentsManager<Dtype>::transform(Caffe::Brew mode, transform_direction transf_dir, const Blob<Dtype>* in_blob, Blob<Dtype>* out_blob, blob_part part) {
  
  int H = in_blob->shape()[2];
  int W = in_blob->shape()[3];
  
  Blob<Dtype>* bank_blob = getOrMakeBank(W,H);   
    
  if(out_blob == NULL) {
    out_blob = temporary_blob_;  
    // reshape also allocates the memory if necessary. If a blob is reshaped to a smaller size, the memory is not deallocated, which is good
    out_blob->ReshapeLike(*in_blob);
  } 
  
  CHECK_EQ(in_blob->shape().size(), out_blob->shape().size());
  for (int i=0; i<in_blob->shape().size(); i++)
    CHECK_EQ(in_blob->shape().at(i), out_blob->shape().at(i)) << "In/Out Shapes differ at dimension " << i;
  
  // actually do the job
  const Dtype* in_data = getBlobPart(in_blob, mode, part);
  const Dtype* bank_data = getBlobPart(bank_blob, mode, BLOB_DATA);
  Dtype* out_data = getMutableBlobPart(out_blob, mode, part);
  
  CBLAS_TRANSPOSE transpose_second = (transf_dir == SPATIAL_TO_SPECTRAL) ? CblasTrans : CblasNoTrans;
  
  if (mode == Caffe::CPU) {
    caffe::caffe_cpu_gemm<float>(CblasNoTrans, transpose_second, in_blob->shape()[0]*in_blob->shape()[1], W*H, W*H, Dtype(1),
        in_data, bank_data, (float)0., out_data);
  } else if (mode == Caffe::GPU) {
    caffe::caffe_gpu_gemm<float>(CblasNoTrans, transpose_second, in_blob->shape()[0]*in_blob->shape()[1], W*H, W*H, Dtype(1),
        in_data, bank_data, (float)0., out_data);
  } else LOG(FATAL) << "Unknown mode " << mode;
  
  return out_blob;
}

template <typename Dtype>
void SpectralComponentsManager<Dtype>::SpectralToSpatial(Caffe::Brew mode, const Blob<Dtype>* in_blob, Blob<Dtype>* out_blob, blob_part part) {
  this->transform(mode, SPECTRAL_TO_SPATIAL, in_blob, out_blob, part);  
}

template <typename Dtype>
void SpectralComponentsManager<Dtype>::SpatialToSpectral(Caffe::Brew mode, const Blob<Dtype>* in_blob, Blob<Dtype>* out_blob, blob_part part) {
  this->transform(mode, SPATIAL_TO_SPECTRAL, in_blob, out_blob, part);  
}

template <typename Dtype>
Blob<Dtype>* SpectralComponentsManager<Dtype>::SpectralToSpatial(Caffe::Brew mode, const Blob<Dtype>* in_blob, blob_part part) {
  return this->transform(mode, SPECTRAL_TO_SPATIAL, in_blob, NULL, part);  
}

template <typename Dtype>
Blob<Dtype>* SpectralComponentsManager<Dtype>::SpatialToSpectral(Caffe::Brew mode, const Blob<Dtype>* in_blob, blob_part part) {
  return this->transform(mode, SPATIAL_TO_SPECTRAL, in_blob, NULL, part);  
}


INSTANTIATE_CLASS(SpectralComponentsManager);

}  // namespace caffe

