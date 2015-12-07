#ifndef COLORFLOW_H__
#define COLORFLOW_H__

extern "C" {

void ColorFlow(const float* input, unsigned char* output, int size, float scale);

void ColorFlow2(const float* input, unsigned char* output, int size, float scale);

}

#endif  // COLORFLOW_H__

