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
  
  Blob<Dtype> *blob = basis_functions_map_[make_pair(W,H)]; // get or create
  
  if(blob == NULL) {
    blob = new Blob<Dtype>();
    // TODO create bank
    // TODO also creat a temporary blob of the same size
  }
  return blob;
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
void SpectralComponentsManager<Dtype>::transform(Caffe::Brew mode, transform_direction transf_dir, const Blob<Dtype>* in_blob, Blob<Dtype>* out_blob) {
  
  int H = in_blob->shape()[2];
  int W = in_blob->shape()[3];
  
  Blob<Dtype>* bank_blob = getOrMakeBank(W,H);   
   
  // actually do the job
  if (mode == Caffe::CPU) {
    const Dtype* in_data = in_blob->cpu_data();
    const Dtype* bank_data = bank_blob->cpu_data();
    Dtype* out_data = out_blob->mutable_cpu_data();
    if (transf_dir == SPATIAL_TO_SPECTRAL) {
      caffe::caffe_cpu_gemm<float>(CblasNoTrans, CblasTrans, in_blob->shape()[0]*in_blob->shape()[1], W*H, W*H, Dtype(1),
          in_data, bank_data, (float)0., out_data);
    } else if (transf_dir == SPECTRAL_TO_SPATIAL) {
      caffe::caffe_cpu_gemm<float>(CblasNoTrans, CblasNoTrans, in_blob->shape()[0]*in_blob->shape()[1], W*H, W*H, Dtype(1),
          in_data, bank_data, (float)0., out_data);
    } else 
      LOG(FATAL) << "Unknown transform_direction " << transf_dir;
  } else if (mode == Caffe::GPU) {
    const Dtype* in_data = in_blob->gpu_data();
    const Dtype* bank_data = bank_blob->gpu_data();
    Dtype* out_data = out_blob->mutable_gpu_data();
    if (transf_dir == SPATIAL_TO_SPECTRAL) {
      caffe::caffe_gpu_gemm<float>(CblasNoTrans, CblasTrans, in_blob->shape()[0]*in_blob->shape()[1], W*H, W*H, Dtype(1),
          in_data, bank_data, (float)0., out_data);
    } else if (transf_dir == SPECTRAL_TO_SPATIAL) {
      caffe::caffe_gpu_gemm<float>(CblasNoTrans, CblasNoTrans, in_blob->shape()[0]*in_blob->shape()[1], W*H, W*H, Dtype(1),
          in_data, bank_data, (float)0., out_data);
    } else 
      LOG(FATAL) << "Unknown transform_direction " << transf_dir;
  } else
    LOG(FATAL) << "Unknown mode " << mode;  
}

template <typename Dtype>
void SpectralComponentsManager<Dtype>::SpectralToSpatial(Caffe::Brew mode, const Blob<Dtype>* in_blob, Blob<Dtype>* out_blob) {
  this->transform(mode, SPECTRAL_TO_SPATIAL, in_blob, out_blob);  
}


INSTANTIATE_CLASS(SpectralComponentsManager);

}  // namespace caffe


//------------

/*
int main(int argc, char** argv) {
  if (argc < 3) {
    std::cout << "Usage: make_fourier_banks FILTER_WIDTH FILTER_HEIGHT" << std::endl;
    return 0;
  }
    
  int W = atoi(argv[1]);
  int H = atoi(argv[2]);
  
  Blob<float> forward_bank(1,W*H,H,W);
  
  float* data = forward_bank.mutable_cpu_data();
  
  float* collage = new float[W*(W+1)*H*(H+1)];
  
  caffe::caffe_set(W*(W+1)*H*(H+1), (float)0., collage);
  
  for (int w=0; w<W; w++)
    for (int h=0; h<H; h++) {
      std::cout << " ===== w=" << w << ", h=" << h << "======== " << std::endl;
      int start_ind = (h * W + w) * H * W; 
      for (int y=0; y<H; y++) {
        for (int x=0; x<W; x++) {
          int ind = start_ind + y * W + x;
          data[ind] = real_dft2(W, w, x, H, h, y);        
        }        
      }
      float norm_coeff = caffe::caffe_cpu_dot(W*H, data + start_ind, data + start_ind);
      norm_coeff = sqrt(norm_coeff);
      std::cout << " >> norm_coeff=" << norm_coeff << std::endl;
      for (int y=0; y<H; y++) {
        for (int x=0; x<W; x++) {
          int ind = start_ind + y * W + x;
          data[ind] /= norm_coeff;
          int collage_ind = (h*(H+1) + y)*W*(W+1) + w*(W+1) + x;
          collage[collage_ind] = (data[ind]+1.)/2.*255.;
          std::cout << data[ind] << " ";
        }
        std::cout << std::endl;
      }
    }
        
  caffe::writePGM(std::string("/misc/lmbraid17/sceneflownet/common/sandbox/fourier.pgm"), (const float*)collage, W*(W+1),H*(H+1));  
  
  float* product = new float[W*H*W*H];
  
  caffe::caffe_cpu_gemm<float>(CblasNoTrans, CblasTrans, W*H, W*H, W*H, (float)100000.,
      data, data, (float)0., product);
  
  caffe::writePGM(std::string("/misc/lmbraid17/sceneflownet/common/sandbox/fourier_product.pgm"), (const float*)product, W*W,H*H);
  
  delete[] collage;
  return 0;
}
*/