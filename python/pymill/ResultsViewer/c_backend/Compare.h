#ifndef COMPARE_H__
#define COMPARE_H__

extern "C" {

/**
 * @brief Compare two two-channel flow images pixelwise and create 
 *        a per-pixel EPE image
 *
 * @param in_a First input flow image
 * @param in_b Second input flow image
 * @param out Output image (must be already allocated)
 * @param width Width of all images (in pixels)
 * @param height Height of all images (in pixels)
 */
void PixelwiseEPE(
      const float* in_a,
      const float* in_b,
      float* out, 
      int width,
      int height);


/**
 * @brief Compare two two-channel flow images pixelwise and create a
 *        flow-difference image
 *
 * @param in_a First input flow image
 * @param in_b Second input flow image
 * @param out Output image (must be already allocated)
 * @param width Width of all images (in pixels)
 * @param height Height of all images (in pixels)
 */
void FlowDelta(
      const float* in_a,
      const float* in_b,
      float* out, 
      int width,
      int height);


/**
 * @brief Compare two single-channel float images pixelwise and create
 *        a difference image
 *
 * @param in_a First input image
 * @param in_b Second input image
 * @param out Output image (must be already allocated)
 * @param width Width of all images (in pixels)
 * @param height Height of all images (in pixels)
 */
void FloatDelta(
      const float* in_a,
      const float* in_b,
      float* out, 
      int width,
      int height);

}

#endif  // COMPARE_H__

