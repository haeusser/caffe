#ifndef CAFFE_DATA_LAYERS_HPP_
#define CAFFE_DATA_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "hdf5.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/filler.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/db.hpp"

#include "lmdb.h"

namespace caffe {
  
/**
 * @brief Apply flow layer
 *
 */
template <typename Dtype>
class ApplyFlowLayer : public Layer<Dtype> {
 public:
  explicit ApplyFlowLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~ApplyFlowLayer() {};
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

 protected:  
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)  { LOG(FATAL) << "ApplyFlowLayer cannot do backward."; return; }
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)  { LOG(FATAL) << "ApplyFlowLayer cannot do backward."; return; }


  Dtype mag_bin_[20];
};
  
/**
 * @brief Abstract Data Augmentation Layer
 *
 */

template <typename Dtype>
class AugmentationLayerBase {
public:
    class tTransMat {
        //tTransMat for Augmentation
        // | 0 2 4 |
        // | 1 3 5 |
    public:
        float t0, t2, t4;
        float t1, t3, t5;

        void toIdentity();

        void leftMultiply(float u0, float u1, float u2, float u3, float u4, float u5);

        void fromCoeff(AugmentationCoeff* coeff,int width,int height,int bottomwidth,int bottomheight);
        
        tTransMat inverse();
    };    

    class tChromaticCoeffs
    {
    public:
        float gamma;
        float brightness;
        float contrast;
        float color[3];

        void fromCoeff(AugmentationCoeff* coeff) { gamma=coeff->gamma(); brightness=coeff->brightness(); contrast=coeff->contrast(); color[0]=coeff->color1(); color[1]=coeff->color2(); color[2]=coeff->color3();  }

        bool needsComputation() { return gamma!=1 || brightness!=0 || contrast!=1 || color[0]!=1 || color[1]!=1 || color[2]!=1; }
    };

    class tChromaticEigenCoeffs
    {
    public:
        float pow_nomean0;
        float pow_nomean1;
        float pow_nomean2;
        float add_nomean0;
        float add_nomean1;
        float add_nomean2;
        float mult_nomean0;
        float mult_nomean1;
        float mult_nomean2;
        float pow_withmean0;
        float pow_withmean1;
        float pow_withmean2;
        float add_withmean0;
        float add_withmean1;
        float add_withmean2;
        float mult_withmean0;
        float mult_withmean1;
        float mult_withmean2;
        float lmult_pow;
        float lmult_add;
        float lmult_mult;
        float col_angle;

        void fromCoeff(AugmentationCoeff* coeff) {
            pow_nomean0=coeff->pow_nomean0();       pow_nomean1=coeff->pow_nomean1();       pow_nomean2=coeff->pow_nomean2();
            add_nomean0=coeff->add_nomean0();       add_nomean1=coeff->add_nomean1();       add_nomean2=coeff->add_nomean2();
            mult_nomean0=coeff->mult_nomean0();     mult_nomean1=coeff->mult_nomean1();     mult_nomean2=coeff->mult_nomean2();
            pow_withmean0=coeff->pow_withmean0();   pow_withmean1=coeff->pow_withmean1();   pow_withmean2=coeff->pow_withmean2();
            add_withmean0=coeff->add_withmean0();   add_withmean1=coeff->add_withmean1();   add_withmean2=coeff->add_withmean2();
            mult_withmean0=coeff->mult_withmean0(); mult_withmean1=coeff->mult_withmean1(); mult_withmean2=coeff->mult_withmean2();
            lmult_pow=coeff->lmult_pow();           lmult_add=coeff->lmult_add();           lmult_mult=coeff->lmult_mult();
            col_angle=coeff->col_angle(); }

        bool needsComputation() {
            return pow_nomean0!=1    || pow_nomean1!=1    || pow_nomean2!=1
                || add_nomean0!=0    || add_nomean1!=0    || add_nomean2!=0
                || mult_nomean0!=1   || mult_nomean1!=1   || mult_nomean2!=1
                || pow_withmean0!=1  || pow_withmean1!=1  || pow_withmean2!=1
                || add_withmean0!=0  || add_withmean1!=0  || add_withmean2!=0
                || mult_withmean0!=1 || mult_withmean1!=1 || mult_withmean2!=1
                || lmult_pow!=1      || lmult_add!=0      || lmult_mult!=1
                || col_angle!=0; }
    };

    class tEffectCoeffs
    {
    public:
        float fog_amount;
        float fog_size;
        float motion_blur_angle;
        float motion_blur_size;
        float shadow_nx;
        float shadow_ny;
        float shadow_distance;
        float shadow_strength;
        float noise;

        void fromCoeff(AugmentationCoeff* coeff) { fog_amount=coeff->fog_amount(); fog_size=coeff->fog_size(); motion_blur_angle=coeff->motion_blur_angle(); motion_blur_size=coeff->motion_blur_size(); shadow_nx=cos(coeff->shadow_angle()); shadow_ny=sin(coeff->shadow_angle()); shadow_distance=coeff->shadow_distance(); shadow_strength=coeff->shadow_strength(); noise=coeff->noise();}
        bool needsComputation() { return (fog_amount!=0 && fog_size!=0) || motion_blur_size>0 || shadow_strength>0 || noise>0; }
    };

    class tChromaticEigenSpace
    {
    public:
        // Note: these need to be floats for the CUDA atomic operations to work
        float mean_eig [3];
        float mean_rgb[3];
        float max_abs_eig[3];
        float max_rgb[3];
        float min_rgb[3];
        float max_l;
        float eigvec [9];
    };

    void clear_spatial_coeffs(AugmentationCoeff& coeff);
    void generate_spatial_coeffs(const AugmentationParameter& aug, AugmentationCoeff& coeff, Dtype discount_coeff);
    void generate_valid_spatial_coeffs(const AugmentationParameter& aug, AugmentationCoeff& coeff, Dtype discount_coeff,
                                       int width, int height, int cropped_width, int cropped_height, int max_num_tries = 50); 

    void clear_chromatic_coeffs(AugmentationCoeff& coeff);
    void generate_chromatic_coeffs(const AugmentationParameter& aug, AugmentationCoeff& coeff, Dtype discount_coeff);

    void clear_chromatic_eigen_coeffs(AugmentationCoeff& coeff);
    void generate_chromatic_eigen_coeffs(const AugmentationParameter& aug, AugmentationCoeff& coeff, Dtype discount_coeff);

    void clear_effect_coeffs(AugmentationCoeff& coeff);
    void generate_effect_coeffs(const AugmentationParameter& aug, AugmentationCoeff& coeff, Dtype discount_coeff);

    void clear_all_coeffs(AugmentationCoeff& coeff);
    void clear_defaults(AugmentationCoeff& coeff);

    void coeff_to_array(const AugmentationCoeff& coeff, Dtype* out);
    void array_to_coeff(const Dtype* in, AugmentationCoeff& coeff);
    void add_coeff_to_array(const AugmentationCoeff& coeff_in, Dtype* out_params);
};





/**
 * @brief Augmentation Parameter Extraction Layer
 *
 */
template <typename Dtype>
class AugParamExtractionLayer : public AugmentationLayerBase<Dtype>, public Layer<Dtype> {
 public:
  explicit AugParamExtractionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~AugParamExtractionLayer() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

 protected:
  
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)  { LOG(FATAL) << "AugParamExtractionLayer cannot do backward."; return; }
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)  { LOG(FATAL) << "AugParamExtractionLayer cannot do backward."; return; }
      

  shared_ptr<SyncedMemory> coeff_matrices1_;
  Blob<Dtype> all_coeffs1_;
  
  AugParamExtractionParameter_ExtractParam extract_param_;
  float multiplier_;
  
  int num_params_;
};

/**
 * @brief Provides base for data layers that feed blobs to the Net.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class BaseDataLayer : public Layer<Dtype> {
 public:
  explicit BaseDataLayer(const LayerParameter& param);
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden except by the BasePrefetchingDataLayer.
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // Data layers should be shared by multiple solvers in parallel
  virtual inline bool ShareInParallel() const { return true; }
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

 protected:
  TransformationParameter transform_param_;
  shared_ptr<DataTransformer<Dtype> > data_transformer_;
  bool output_labels_;
};

template <typename Dtype>
class Batch {
 public:
  Blob<Dtype> data_, label_;
};

template <typename Dtype>
class BasePrefetchingDataLayer :
    public BaseDataLayer<Dtype>, public InternalThread {
 public:
  explicit BasePrefetchingDataLayer(const LayerParameter& param);
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden.
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  // Prefetches batches (asynchronously if to GPU memory)
  static const int PREFETCH_COUNT = 3;

 protected:
  virtual void InternalThreadEntry();
  virtual void load_batch(Batch<Dtype>* batch) = 0;

  Batch<Dtype> prefetch_[PREFETCH_COUNT];
  BlockingQueue<Batch<Dtype>*> prefetch_free_;
  BlockingQueue<Batch<Dtype>*> prefetch_full_;

  Blob<Dtype> transformed_data_;
};

template <typename Dtype>
class DataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit DataLayer(const LayerParameter& param);
  virtual ~DataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // DataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "Data"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void load_batch(Batch<Dtype>* batch);

  DataReader reader_;
};

/**
 * @brief Provides data to the Net generated by a Filler.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class DummyDataLayer : public Layer<Dtype> {
 public:
  explicit DummyDataLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // Data layers should be shared by multiple solvers in parallel
  virtual inline bool ShareInParallel() const { return true; }
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual inline const char* type() const { return "DummyData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

  vector<shared_ptr<Filler<Dtype> > > fillers_;
  vector<bool> refill_;
};

/**
 * @brief FloatWriterLayer writes float3 files
 *
 */
template <typename Dtype>
class FloatWriterLayer : public Layer<Dtype> {
 public:
  explicit FloatWriterLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~FloatWriterLayer() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  
  virtual inline const char* type() const { return "FloatWriter"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
 protected:

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
};

/**
 * @brief FLOWriterLayer writes FLO (flow) files
 *
 */
template <typename Dtype>
class FLOWriterLayer : public Layer<Dtype> {
 public:
  explicit FLOWriterLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~FLOWriterLayer() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  
  virtual inline const char* type() const { return "FLOWriter"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
 protected:

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
};

/**
 * @brief Optical Flow Augmentation Layer
 *
 */
template <typename Dtype>
class FlowAugmentationLayer : public AugmentationLayerBase<Dtype>, public Layer<Dtype> {
 public:
  explicit FlowAugmentationLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~FlowAugmentationLayer() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline bool AllowBackward() const { LOG(WARNING) << "FlowAugmentationLayer does not do backward."; return false; }
  

 protected:
  
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)  { LOG(FATAL) << "FlowAugmentationLayer cannot do backward."; return; }
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)  { LOG(FATAL) << "FlowAugmentationLayer cannot do backward."; return; }
      

  shared_ptr<SyncedMemory> coeff_matrices1_;
  shared_ptr<SyncedMemory> coeff_matrices2_;
  Blob<Dtype> all_coeffs1_;
  Blob<Dtype> all_coeffs2_;
  
  Blob<Dtype> test_coeffs_;
  
  int cropped_height_;
  int cropped_width_;
  int num_params_;
};

/**
 * @brief Generate Augmentation Parameters Layer
 *
 */
template <typename Dtype>
class GenerateAugmentationParametersLayer : public AugmentationLayerBase<Dtype>, public Layer<Dtype> {
 public:
  explicit GenerateAugmentationParametersLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~GenerateAugmentationParametersLayer() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline bool AllowBackward() const { LOG(WARNING) << "GenerateAugmentationParametersLayer does not do backward."; return false; }
  
 protected:
  
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)  { LOG(FATAL) << "GenerateAugmentationParametersLayer cannot do backward."; return; }
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)  { LOG(FATAL) << "GenerateAugmentationParametersLayer cannot do backward."; return; }
      
    
  int num_params_; 
  int num_iter_;
  int cropped_width_;
  int cropped_height_;
  int bottomwidth_;
  int bottomheight_;
  int num_;
  AugmentationParameter aug_;
  CoeffScheduleParameter discount_coeff_schedule_;
  std::string mode_;
};


/**
 * @brief Provides data to the Net from HDF5 files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class HDF5DataLayer : public Layer<Dtype> {
 public:
  explicit HDF5DataLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~HDF5DataLayer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // Data layers should be shared by multiple solvers in parallel
  virtual inline bool ShareInParallel() const { return true; }
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual inline const char* type() const { return "HDF5Data"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void LoadHDF5FileData(const char* filename);

  std::vector<std::string> hdf_filenames_;
  unsigned int num_files_;
  unsigned int current_file_;
  hsize_t current_row_;
  std::vector<shared_ptr<Blob<Dtype> > > hdf_blobs_;
  std::vector<unsigned int> data_permutation_;
  std::vector<unsigned int> file_permutation_;
};

/**
 * @brief Write blobs to disk as HDF5 files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class HDF5OutputLayer : public Layer<Dtype> {
 public:
  explicit HDF5OutputLayer(const LayerParameter& param)
      : Layer<Dtype>(param), file_opened_(false) {}
  virtual ~HDF5OutputLayer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // Data layers should be shared by multiple solvers in parallel
  virtual inline bool ShareInParallel() const { return true; }
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual inline const char* type() const { return "HDF5Output"; }
  // TODO: no limit on the number of blobs
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 0; }

  inline std::string file_name() const { return file_name_; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void SaveBlobs();

  bool file_opened_;
  std::string file_name_;
  hid_t file_id_;
  Blob<Dtype> data_blob_;
  Blob<Dtype> label_blob_;
};

/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class ImageDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit ImageDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~ImageDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ImageData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);

  vector<std::pair<std::string, int> > lines_;
  int lines_id_;
};


/**
 * @brief ImgWriterLayer writes PPMs and PGMs
 *
 */
template <typename Dtype>
class ImgWriterLayer : public Layer<Dtype> {
 public:
  explicit ImgWriterLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~ImgWriterLayer() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  
  virtual inline const char* type() const { return "ImgWriter"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
 protected:

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
};

/**
 * @brief Phils Data Layer
 *
 */
template <typename Dtype>
void* PhilDataLayerPrefetch(void* layer_pointer);

template <typename Dtype>
class PhilDataLayer : public Layer<Dtype> {
  // The function used to perform prefetching.
  friend void* PhilDataLayerPrefetch<Dtype>(void* layer_pointer);

 public:
  explicit PhilDataLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~PhilDataLayer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline bool ShareInParallel() const { return true; }
  
  virtual inline const char* type() const { return "PhilData"; }
  
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

  virtual void CreatePrefetchThread();
  virtual void JoinPrefetchThread();
  virtual unsigned int PrefetchRand();
  
  virtual void generateRandomPermutation(int seed, int block_size = 0);

  shared_ptr<Caffe::RNG> prefetch_rng_;

  // LMDB
  MDB_env* mdb_env_;
  MDB_dbi mdb_dbi_;
  MDB_txn* mdb_txn_;
  MDB_cursor* mdb_cursor_;
  MDB_val mdb_key_, mdb_value_;

  int database_entries_;
  int datum_index_;
  int range_start_;
  int range_end_;
  int range_size_;
  
  int preselection_label_;
  
  std::vector<int> permutation_vector_;
  
  vector<int> slice_point_;
  vector<int> channel_encoding_;
  
  int datum_channels_;
  int datum_height_;
  int datum_width_;
  int datum_size_;
  pthread_t thread_;
  shared_ptr<Blob<Dtype> > prefetch_label_;
  //shared_ptr<Blob<Dtype> > prefetch_data_;
  vector<shared_ptr<Blob<Dtype> > > prefetch_data_blobs_;
  
  Blob<Dtype> data_mean_;
  bool output_labels_;
  
  int iter_;
};




/**
 * @brief Phil Data Augmentation Layer
 *
 */
template <typename Dtype>
class PhilDataAugmentationLayer : public AugmentationLayerBase<Dtype>, public Layer<Dtype> {
public:
    explicit PhilDataAugmentationLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {}
    virtual ~PhilDataAugmentationLayer() {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);
    virtual void adjust_blobs(const LayerParameter& source_layer);
    virtual inline bool AllowBackward() const { LOG(WARNING) << "PhilDataAugmentationLayer does not do backward."; return false; }

protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);

    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)  { LOG(FATAL) << "PhilDataAugmentationLayer cannot do backward."; return; }
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)  { LOG(FATAL) << "PhilDataAugmentationLayer cannot do backward."; return; }
                                
    int cropped_height_;
    int cropped_width_;
    bool do_cropping_;
    int num_params_;

    bool output_params_;
    bool input_params_;
    Blob<Dtype> all_coeffs_;
    shared_ptr<SyncedMemory> coeff_matrices_;
    shared_ptr<SyncedMemory> coeff_chromatic_;
    shared_ptr<SyncedMemory> coeff_chromatic_eigen_;
    shared_ptr<SyncedMemory> coeff_effect_;
    shared_ptr<SyncedMemory> coeff_colors_;
    shared_ptr<SyncedMemory> chromatic_eigenspace_;

    shared_ptr<SyncedMemory> noise_;

    AugmentationParameter aug_;
    CoeffScheduleParameter discount_coeff_schedule_;
    Blob<Dtype> ones_;
    Blob<Dtype> pixel_rgb_mean_from_proto_;
};



/**
 * @brief Provides data to the Net from memory.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class MemoryDataLayer : public BaseDataLayer<Dtype> {
 public:
  explicit MemoryDataLayer(const LayerParameter& param)
      : BaseDataLayer<Dtype>(param), has_new_data_(false) {}
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MemoryData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

  virtual void AddDatumVector(const vector<Datum>& datum_vector);
  virtual void AddMatVector(const vector<cv::Mat>& mat_vector,
      const vector<int>& labels);

  // Reset should accept const pointers, but can't, because the memory
  //  will be given to Blob, which is mutable
  void Reset(Dtype* data, Dtype* label, int n);
  void set_batch_size(int new_size);

  int batch_size() { return batch_size_; }
  int channels() { return channels_; }
  int height() { return height_; }
  int width() { return width_; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  int batch_size_, channels_, height_, width_, size_;
  Dtype* data_;
  Dtype* labels_;
  int n_;
  size_t pos_;
  Blob<Dtype> added_data_;
  Blob<Dtype> added_label_;
  bool has_new_data_;
};

/**
 * @brief Provides data to the Net from windows of images files, specified
 *        by a window data file.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class WindowDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit WindowDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~WindowDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "WindowData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  virtual unsigned int PrefetchRand();
  virtual void load_batch(Batch<Dtype>* batch);

  shared_ptr<Caffe::RNG> prefetch_rng_;
  vector<std::pair<std::string, vector<int> > > image_database_;
  enum WindowField { IMAGE_INDEX, LABEL, OVERLAP, X1, Y1, X2, Y2, NUM };
  vector<vector<float> > fg_windows_;
  vector<vector<float> > bg_windows_;
  Blob<Dtype> data_mean_;
  vector<Dtype> mean_values_;
  bool has_mean_file_;
  bool has_mean_values_;
  bool cache_images_;
  vector<std::pair<std::string, Datum > > image_database_cache_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYERS_HPP_
