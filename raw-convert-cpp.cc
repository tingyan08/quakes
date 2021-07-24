#include <stdio.h>
#include <stdlib.h>
#include <opencv2/imgproc.hpp>
#include <algorithm>

extern "C"
{
#include "raw-convert-cpp.h"
}

extern "C"
bool debayer( // out
              uint8_t* out_rgb,

              // in
              const uint8_t* raw_bits,
              int width, int height )
{
    cv::Mat in (height, width, CV_8U,   (void*)raw_bits);
    cv::Mat out(height, width, CV_8UC3, out_rgb);

    cvtColor(in, out, cv::COLOR_BayerRG2RGB, 3);
    return true;
}

extern "C"
uint8_t ith_element( uint8_t* buf, int N, int i )
{
    std::nth_element(&buf[0], &buf[i], &buf[N]);
    return buf[i];
}

extern "C"
void opencv_disable_threading(void)
{
    cv::setNumThreads(0);
}

