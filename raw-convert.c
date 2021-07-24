#include <stdio.h>
#include <stdlib.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <errno.h>
#include <inttypes.h>
#include <string.h>

#include "raw-convert-cpp.h"

#define MSG(fmt, ...) \
    fprintf(stderr, "%s(%d): " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)

__attribute__((visibility("default")))
bool raw_convert( // out (width*height*3 bytes)
                  uint8_t* out_rgb,

                  // in
                  const char* filename_raw, int width, int height,
                  uint32_t saturate_recip)
{
    bool result = false;

    // I'm going to parallelize this thing on a higher level, so I'd like these
    // INNER calls to run sequentially. I don't want the user to ask 20-way
    // parallelization, then get 20 parallel raw_convert() calls running, and
    // then have some internal opencv something spawn 20 children each.
    // 20*20=400 threads is not helpful
    opencv_disable_threading();

    struct stat st;
    if(0 != stat(filename_raw, &st))
    {
        MSG("stat(%s) failed. Does this file exist?", filename_raw);
        return false;
    }

    int datasize_expected = width*height;

    if(st.st_size <= datasize_expected)
    {
        MSG("I expected %d bytes of pixel data, but the given file only has %d bytes in it. That is too small",
            datasize_expected,
            (int)st.st_size);
        return false;
    }

    uint8_t* raw_bits = MAP_FAILED;
    int fd = open(filename_raw, O_RDONLY);
    uint8_t* buf = NULL;

    if(fd < 0)
    {
        MSG("open(%s) failed", filename_raw);
        goto done;
    }

    raw_bits = mmap(NULL, st.st_size,
                    PROT_READ, MAP_SHARED,
                    fd, 0);
    if(raw_bits == MAP_FAILED)
    {
        MSG("mmap(%s) failed: %s", filename_raw,
            strerror(errno));
        goto done;
    }

    if(!debayer(out_rgb, &raw_bits[st.st_size - datasize_expected],
                width, height))
    {
        MSG("debayer(%s) failed", filename_raw);
        goto done;
    }

    // I decimate heavily when computing the mean and percentiles.
#define DECIMATION_LOG2 3
#define DECIMATION (1U << DECIMATION_LOG2)
    if( width & (DECIMATION-1) )
    {
        MSG("width (%d) assumed to evently divide the decimation factor (%d), but it doesnt. Look through the code",
            width, DECIMATION);
        goto done;
    }
    if( height & (DECIMATION-1) )
    {
        MSG("height (%d) assumed to evently divide the decimation factor (%d), but it doesnt. Look through the code",
            height, DECIMATION);
        goto done;
    }

    uint64_t N = (width >> DECIMATION_LOG2) * (height >> DECIMATION_LOG2);
    buf = malloc(N);
    if(buf == NULL)
    {
        MSG("Couldn't malloc(%"PRId64")", N);
        goto done;
    }

    // I process each channel separately
    for(int i_channel=0; i_channel<3; i_channel++)
    {
        // First, grayworld. This just means "scale the intensities to
        // move the mean to the center"

        // Compute the sum (in order to compute the mean later). And copy the
        // decimated data into a separate buffer for the later percentile
        // computation. That percentile computation is inefficient, but probably
        // good-enough
        uint64_t sum = 0;
        int ibuf = 0;
        for(int i=0; i<height; i += DECIMATION)
            for(int j=0; j<width; j += DECIMATION)
            {
                buf[ibuf] = out_rgb[(i*width + j)*3 + i_channel];
                sum += buf[ibuf];
                ibuf++;
            }

        // I will need to compute x * 128/mean, saturating the overflows, and
        // hopefully being fast. The denominator is always the same, so I use
        // higher-order multiplication to do the division. I tried libdivide,
        // but it wasn't as fast
        //
        // y = x*128/(sum/N)
        //   = x*N*128*(1 << (3*8))/sum >> (3*8)
        uint64_t k0 = (N*128ULL*(1ULL << (3*8)))/sum;
        uint8_t grayworld(uint8_t x)
        {
            uint64_t y = (k0 * (uint64_t)(x)) >> (3*8);
            if( y >= 255) return 255;
            return y;
        }

        for(int i=0; i<height; i++)
            for(int j=0; j<width; j++)
            {
                uint8_t* x = &out_rgb[(i*width + j)*3 + i_channel];
                *x = grayworld(*x);
            }

        // The input was written back to the array. The mean was scaled
        // to 128. Grayworld is done

        // I linearly stretch the intensities such that the histogram
        // covers my full range, sacrificing some stragglers at the low
        // and high ends

        // I find the imin-th element. This re-orders the array, so the imin-th
        // element is correct, and everything leading to it is < xmin and
        // everything past is > xmin
        uint32_t imin = N/saturate_recip;
        uint8_t  xmin = ith_element(buf, N, imin);

        // I find the imax-th element. Since the array is already partially
        // ordered, I don't include the data on the bottom end
        uint32_t imax = N-1 - imin;
        uint8_t xmax = ith_element(&buf[imin], N-imin, imax-imin);

        if(xmin == xmax)
        {
            MSG("saturating the lower and upper 1/%d results in xmin = xmax = %d. Cut back on the saturation",
                saturate_recip,
                xmin);
            goto done;
        }

        // xmin and xmax came form the pre-grayworld buffer, so I apply the
        // correction here
        xmin = grayworld(xmin);
        xmax = grayworld(xmax);

        // I'm going to be computing ... / (xmax-xmin) a lot. Use higher-order
        // multiplication to make that fast. I tried libdivide as well, but it
        // was slower
        //   y =
        //   (x-xmin) * 255/(xmax-xmin) =
        //   (x-xmin) * 255*(1 << (6*8))/(xmax-xmin) >> (6*8)
        uint64_t k1 = (255ULL*(1ULL << (6*8)))/(uint64_t)(xmax-xmin);

        for(int i=0; i<height; i++)
            for(int j=0; j<width; j++)
            {
                uint8_t* x = &out_rgb[(i*width + j)*3 + i_channel];
                if(     *x <= xmin) *x = 0;
                else if(*x >= xmax) *x = 255;
                else
                    *x = (k1 * (uint64_t)(*x - xmin)) >> (6*8);
            }
    }

    result = true;

 done:
    if(raw_bits != MAP_FAILED)
        munmap(raw_bits, datasize_expected);
    if(fd >= 0)
        close(fd);
    free(buf);

    return result;
}
