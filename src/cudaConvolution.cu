/*
 .----------------------------------------------------------.
 |  Optimized StrayLight implementation in C++/OpenMP/CUDA  |
 |      Task 2.2 of "TASTE Maintenance and Evolutions"      |
 |           P12001 - PFL-PTE/JK/ats/412.2012i              |
 |                                                          |
 |  Contact Point:                                          |
 |           Thanassis Tsiodras, Dr.-Ing.                   |
 |           ttsiodras@semantix.gr / ttsiodras@gmail.com    |
 |           NeuroPublic S.A.                               |
 |                                                          |
 |   Licensed under the GNU Lesser General Public License,  |
 |   details here:  http://www.gnu.org/licenses/lgpl.html   |
 `----------------------------------------------------------'

*/

#include <stdio.h>

#include "configStraylight.h"
#include "utilities.h"

#define THREADS_PER_BLOCK 256

#define inX IMGWIDTH
#define inY IMGHEIGHT
#define kX  KERNEL_SIZE
#define kY  KERNEL_SIZE

#ifdef USE_DOUBLEPRECISION

// CUDA doesn't have double-precision textures, but we emulate them
// using textures of int2 - and a conversion function.

#include "sm_13_double_functions.h"

texture<int2, 1, cudaReadModeElementType>       g_cudaMainMatrixTexture;
texture<int2, 1, cudaReadModeElementType>       g_cudaMainKernelTexture;

__inline__ __device__ double fetch_double(const texture<int2, 1>& t, int i)
{
    int2 v = tex1Dfetch(t,i);
    return __hiloint2double(v.y, v.x);
}

#else // USE_DOUBLEPRECISION

texture<float1, 1, cudaReadModeElementType>     g_cudaMainMatrixTexture;
texture<float1, 1, cudaReadModeElementType>     g_cudaMainKernelTexture;

#endif // USE_DOUBLEPRECISION

__global__ void DoWork(fp *cudaOutputImage/*, int inX, int inY, int kX, int kY*/)
{
    unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= inX*inY)
	return;

    int i = idx / inX;
    int j = idx % inX;

    fp sum=0.;
    for(int k=0; k<kY; k++) {
        int ofsY = i+k;
        if (ofsY>=inY) ofsY -= inY;
        unsigned input_image_corrected_line_Offset = inX*ofsY;
        unsigned psf_high_res_line_Offset = kX*k;

        #pragma unroll
        for(int l=0; l<kX; l++) {
            int ofsX = j+l;
            //if (__builtin_expect(ofsX>=inX,0)) ofsX -= inX;
            if (ofsX>=inX)
                ofsX -= inX;

            //sum += input_image_corrected_line[ofsX]*psf_high_res_line[l];
#ifdef USE_DOUBLEPRECISION
            sum += 
                fetch_double(g_cudaMainMatrixTexture, input_image_corrected_line_Offset + ofsX)
                *
                fetch_double(g_cudaMainKernelTexture, psf_high_res_line_Offset + l);
            
#else
            sum += 
                tex1Dfetch(g_cudaMainMatrixTexture, input_image_corrected_line_Offset + ofsX).x
                *
                tex1Dfetch(g_cudaMainKernelTexture, psf_high_res_line_Offset + l).x;
#endif
        }
    }
    int outOfsX = j+kX/2;
    if (outOfsX>=inX) outOfsX -= inX;

    int sY = i+kY/2;
    if (sY >= inY) sY -= inY;
    cudaOutputImage[sY*inX + outOfsX] = sum;
}

void cudaConvolution(
    fp *cudaMainMatrix, fp *cudaMainKernel, fp *cudaOutputImage
    /*, int inX, int inY, int kX, int kY*/)
{
#ifdef USE_DOUBLEPRECISION
    cudaChannelFormatDesc channel1desc = cudaCreateChannelDesc<int2>();
    cudaChannelFormatDesc channel2desc = cudaCreateChannelDesc<int2>();
#else
    cudaChannelFormatDesc channel1desc = cudaCreateChannelDesc<float1>();
    cudaChannelFormatDesc channel2desc = cudaCreateChannelDesc<float1>();
#endif
    cudaBindTexture(NULL, &g_cudaMainMatrixTexture, cudaMainMatrix, &channel1desc, inX*inY*sizeof(fp));
    cudaBindTexture(NULL, &g_cudaMainKernelTexture, cudaMainKernel, &channel2desc, kX*kY*sizeof(fp));

    int blocks = (inX*inY + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
    DoWork<<< blocks, THREADS_PER_BLOCK >>>(cudaOutputImage/*, inX, inY, kX, kY*/);
    cudaError_t error = cudaThreadSynchronize();
    if(error != cudaSuccess) {
        cudaError_t error = cudaGetLastError();
	debug_printf(
            LVL_PANIC, "CUDA error: %s\n", cudaGetErrorString(error));
    }

    cudaUnbindTexture(&g_cudaMainKernelTexture);
    cudaUnbindTexture(&g_cudaMainMatrixTexture);
}

