#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

namespace tolerance_layer {
  
  template <typename Dtype>
  __global__ void ToleranceTransform(const int nthreads,
      const int num, const int channels, const int height, const int width,
      const int tol_rad, 
      const Dtype* in_gt, const Dtype* in_data, 
      Dtype* out_data) {
    
    CUDA_KERNEL_LOOP(index, nthreads) {
      int idx_w = index % width; //width
      int idx_h = (index / width) % height; // height
      //int idx_c = (index / width / height) % channels; // channels
      
      
      Dtype min_diff = Dtype(1e10);
      Dtype cur_data = in_data[index];
      Dtype min_diff_val = in_gt[index];
      
      for(int j = -tol_rad; j <= tol_rad; j++) {
        for(int i = -tol_rad; i <= tol_rad; i++) {
          //printf("[%d] (i,j)=(%d,%d)\n", index, i, j);
          if(idx_w + i >= 0 && idx_w + i < width && idx_h + j >= 0 && idx_h + j < height) {
            int in_gt_pos = index + (j*width) + i;
            
            Dtype gt_val = in_gt[in_gt_pos];
            Dtype diff = fabsf(gt_val - cur_data);
            //printf("[%d] %f<%f -> %f ?\n", index, diff, min_diff, gt_val);
            if(diff < min_diff) {
              min_diff = diff;
              min_diff_val = gt_val;
            }
          }
        }
      }
      out_data[index] = min_diff_val; 
    }
  }

}

template <typename Dtype>
void ToleranceLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  
  const int count = bottom[0]->count();
  
  tolerance_layer::ToleranceTransform<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    count,
    num, channels, height, width,
    tolerance_radius_,
    bottom[0]->gpu_data(), bottom[1]->gpu_data(),
    top[0]->mutable_gpu_data()
    );
}

template <typename Dtype>
void ToleranceLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // No gradient:
  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      caffe_gpu_set(bottom[i]->count(), Dtype(0),
                    bottom[i]->mutable_gpu_data());
    }
  }
}
INSTANTIATE_LAYER_GPU_FUNCS(ToleranceLayer);

}  // namespace caffe
