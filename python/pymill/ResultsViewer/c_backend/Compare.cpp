
#include <cmath>
#include "Compare.h"


/**
 * @brief Endpoint error between two 2d vectors
 *
 * @param a First vector
 * @param b Second vector
 *
 * @returns The Euclidean (=L2) distance between a and b, i.e. 
 *          ||a^2-b^2||_2
 */
float EPE(const float* a, const float* b)
{
  return sqrt((a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]));
}


void PixelwiseEPE(const float* in_a, 
                  const float* in_b,
                  float* out,
                  int width,
                  int height)
{
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const float* inpixel_a = &(in_a[(y*width+x)*2]);
      const float* inpixel_b = &(in_b[(y*width+x)*2]);
      float* outpixel        = &(out[(x*height+y)]);

      outpixel[0] = EPE(inpixel_a, inpixel_b);
    }
  }
}


void FlowDelta(const float* in_a, 
               const float* in_b,
               float* out,
               int width,
               int height)
{
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const float* inpixel_a = &(in_a[(y*width+x)*2]);
      const float* inpixel_b = &(in_b[(y*width+x)*2]);
      float* outpixel        = &(out[ (y*width+x)*2]);

      outpixel[0] = inpixel_a[0] - inpixel_b[0];
      outpixel[1] = inpixel_a[1] - inpixel_b[1];
    }
  }
}


void FloatDelta(const float* in_a, 
                const float* in_b,
                float* out,
                int width,
                int height)
{
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const float& inpixel_a = in_a[y*width+x];
      const float& inpixel_b = in_b[y*width+x];
      float& outpixel        =  out[y*width+x];

      outpixel = std::abs(inpixel_a - inpixel_b);
    }
  }
}


