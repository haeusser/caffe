#include "caffe/caffe.hpp"
#include "caffe/blob.hpp"
#include "caffe/util/output.hpp"
#include <string>
#include "caffe/util/math_functions.hpp"

using caffe::Blob;

float pi = 3.141592653589793238462643383279;

// float real_dft(int N, int n, int x) {
//   if (n == 0 || 2*n == N)
//     return 1.;
//   if (n <= (N-1)/2)
//     return cos(2*pi*(float)(n)*(float)(x)/(float)(N));
//   else
//     return sin(2*pi*(float)(n-N/2)*(float)(x)/(float)(N));
//   
//   return 0.;
// }

float real_dft2(int W, int w, int x, int H, int h, int y) {
  float fx = (float)(x);
  float fy = (float)(y);
  float fw = (float)(w);
  float fh = (float)(h);
  float fH = (float)(H);
  float fW = (float)(W);
  
  if ((2*h <= H && 2*w <= W) || (2*h > H && (w > 0 && 2*w < W)))
    return cos(2*pi*(fw*fx/fW + fh*fy/fH));
  if (2*h > H && (w == 0 || 2*w == W))
    return sin(2*pi*(fw*fx/fW + (fh-(float)(H/2))*fy/fH));
  if (2*w > W)
    return sin(2*pi*((fw-(float)(W/2))*fx/fW + fh*fy/fH));
  
  return 0.;
}

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

