#pragma once

#include <stdbool.h>
#include <stdint.h>

bool raw_convert( // out (width*height*3 bytes)
                  uint8_t* out_rgb,

                  // in
                  const char* filename_raw, int width, int height,
                  bool do_grayworld,
                  uint32_t saturate_recip);
