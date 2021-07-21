#pragma once

#include <stdbool.h>
#include <stdint.h>

bool debayer( // out
              uint8_t* out_rgb,

              // in
              const uint8_t* raw_bits,
              int width, int height );

// Efficiently compute the ith sorted element of the given buffer. The buffer is
// reordered
uint8_t ith_element( uint8_t* buf, int N, int i );
