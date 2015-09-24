
#include "CartesianColorWheel.h"
#include "ColorFlow.h"


void ColorFlow(const float* input, unsigned char* output, int size, float scale)
{
  float pix[3] = {0.f, 0.f, 0.f};

  for (int i = 0; i < size; ++i) {
    const float* inpixel = &(input[i*2]);
    unsigned char* outpixel = &(output[i*3]);

    sintelCartesianToRGB(inpixel[0]*scale, inpixel[1]*scale, pix);

    outpixel[0] = (unsigned char)(pix[0]*255.f);
    outpixel[1] = (unsigned char)(pix[1]*255.f);
    outpixel[2] = (unsigned char)(pix[2]*255.f);
  }

}


void cartesianToRGB (float x, float y, float* R, float* G, float* B)
{
  const float Pi = 3.1415926536f;
  float radius = sqrt (x * x + y * y);
  if (radius > 1) radius = 1;
  float phi;
  if (x == 0.0)
    if (y >= 0.0) phi = 0.5 * Pi;
    else phi = 1.5 * Pi;
  else if (x > 0.0)
    if (y >= 0.0) phi = atan (y/x);
    else phi = 2.0 * Pi + atan (y/x);
  else phi = Pi + atan (y/x);
  float alpha, beta;    // weights for linear interpolation
  phi *= 0.5;
  // interpolation between red (0) and blue (0.25 * Pi)
  if ((phi >= 0.0) && (phi < 0.125 * Pi)) {
    beta  = phi / (0.125 * Pi);
    alpha = 1.0 - beta;
    (*R) = (int)(radius * (alpha * 255.0 + beta * 255.0));
    (*G) = (int)(radius * (alpha *   0.0 + beta *   0.0));
    (*B) = (int)(radius * (alpha *   0.0 + beta * 255.0));
  }
  if ((phi >= 0.125 * Pi) && (phi < 0.25 * Pi)) {
    beta  = (phi-0.125 * Pi) / (0.125 * Pi);
    alpha = 1.0 - beta;
    (*R) = (int)(radius * (alpha * 255.0 + beta *  64.0));
    (*G) = (int)(radius * (alpha *   0.0 + beta *  64.0));
    (*B) = (int)(radius * (alpha * 255.0 + beta * 255.0));
  }
  // interpolation between blue (0.25 * Pi) and green (0.5 * Pi)
  if ((phi >= 0.25 * Pi) && (phi < 0.375 * Pi)) {
    beta  = (phi - 0.25 * Pi) / (0.125 * Pi);
    alpha = 1.0 - beta;
    (*R) = (int)(radius * (alpha *  64.0 + beta *   0.0));
    (*G) = (int)(radius * (alpha *  64.0 + beta * 255.0));
    (*B) = (int)(radius * (alpha * 255.0 + beta * 255.0));
  }
  if ((phi >= 0.375 * Pi) && (phi < 0.5 * Pi)) {
    beta  = (phi - 0.375 * Pi) / (0.125 * Pi);
    alpha = 1.0 - beta;
    (*R) = (int)(radius * (alpha *   0.0 + beta *   0.0));
    (*G) = (int)(radius * (alpha * 255.0 + beta * 255.0));
    (*B) = (int)(radius * (alpha * 255.0 + beta *   0.0));
  }
  // interpolation between green (0.5 * Pi) and yellow (0.75 * Pi)
  if ((phi >= 0.5 * Pi) && (phi < 0.75 * Pi)) {
    beta  = (phi - 0.5 * Pi) / (0.25 * Pi);
    alpha = 1.0 - beta;
    (*R) = (int)(radius * (alpha * 0.0   + beta * 255.0));
    (*G) = (int)(radius * (alpha * 255.0 + beta * 255.0));
    (*B) = (int)(radius * (alpha * 0.0   + beta * 0.0));
  }
  // interpolation between yellow (0.75 * Pi) and red (Pi)
  if ((phi >= 0.75 * Pi) && (phi <= Pi)) {
    beta  = (phi - 0.75 * Pi) / (0.25 * Pi);
    alpha = 1.0 - beta;
    (*R) = (int)(radius * (alpha * 255.0 + beta * 255.0));
    (*G) = (int)(radius * (alpha * 255.0 + beta *   0.0));
    (*B) = (int)(radius * (alpha * 0.0   + beta *   0.0));
  }
  if ((*R)<0) R=0;
  if ((*G)<0) G=0;
  if ((*B)<0) B=0;
  if ((*R)>255) (*R)=255;
  if ((*G)>255) (*G)=255;
  if ((*B)>255) (*B)=255;
}


void ColorFlow2(const float* input, unsigned char* output, int size, float scale)
{
  float pix[3] = {0.f, 0.f, 0.f};

  for (int i = 0; i < size; ++i) {
    const float* inpixel = &(input[i*2]);
    unsigned char* outpixel = &(output[i*3]);

    cartesianToRGB(inpixel[0]*scale, inpixel[1]*scale, 
                   &pix[0], &pix[1], &pix[2]);

    outpixel[0] = (unsigned char)(pix[0]);
    outpixel[1] = (unsigned char)(pix[1]);
    outpixel[2] = (unsigned char)(pix[2]);
  }
}



