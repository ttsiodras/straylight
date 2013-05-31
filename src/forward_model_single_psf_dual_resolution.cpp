/*
 .----------------------------------------------------------.
 |  Optimized StrayLight implementation in C++/OpenMP/CUDA  |
 |      Task 2.2 of "TASTE Maintenance and Evolutions"      |
 |           P12001 - PFL-PTE/JK/ats/412.2012i              |
 |                                                          |
 |  Contact Point:                                          |
 |           Thanassis Tsiodras, Dr.-Ing.                   |
 |           ttsiodras@gmail.com / ttsiodras@semantix.gr    |
 |           NeuroPublic S.A.                               |
 |                                                          |
 |   Licensed under the GNU Lesser General Public License,  |
 |   details here:  http://www.gnu.org/licenses/lgpl.html   |
 `----------------------------------------------------------'

*/

// Comments from the original IDL content start like this: '//#'
// Normal C++ comments are used for porting-related information.

#include <iostream>
#include <memory>
#include <fstream>
#include <map>

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

// General configuration parameters are in this header.
#include "configStraylight.h"

// FFTW3 is used for very fast FFT / invFFT
#include <fftw3.h>

// Utility functions containing parts of the algorithm
#include "harvey_psf.h"
#include "tis_surface_scattering_harvey.h"
#include "particulate_contamination_harvey_psf.h"

#include "utilities.h"

// Convolution specific section: can be done via 
//
// - our custom C++ code using Eigen
// - our custom CUDA kernel
// - our custom cache-optimized C++ code
//
// The USE_EIGEN and USE_CUDA_GPU #ifdefs (set via ./configure) 
// decide how convolution is done...

#ifdef USE_CUDA_GPU

////////////////////////////////////////////////////////////////////////////
//
// CUDA can also be used to perform an even faster convolution.
// Tested with an 150$ GTX 650 Ti with 1GB GDDR5 memory, it runs
// 5 times faster than Eigen's SSE code running in all 6 cores of
// a Phenom II X6 1090T.
//
////////////////////////////////////////////////////////////////////////////

#include <cuda.h>
#include <cuda_runtime.h>

// API-wrapping macro, so we easily check all CUDA API invocations
#define SAFE(call) do {                                                      \
    cudaError_t err = call;                                                  \
    if (cudaSuccess != err)                                                  \
        debug_printf(                                                        \
            LVL_PANIC,                                                       \
            "Cuda driver error %x in file '%s' in line %i.\n",               \
            err, __FILE__, __LINE__ );                                       \
} while (0)

// For speed reasons, we need to use "pinned" memory buffers, 
// that are copied to/from GPU memory via fast DMA transfers.
// We therefore define the two special types of arrays
// (m2dSpecialImage and m2dSpecialPSF) using the cudaMallocHost API.
template <int height, int width, bool useCudaStaticMem=false>
struct m2dCuda {
    static const int _width = width;
    static const int _height = height;
    fp *_pData;
    m2dCuda(int y, int x) {
        if (y != height || x != width) {
            debug_printf(
                LVL_PANIC,
                "forward_model_single_psf_dual_resolution violated the contract!\n"
                "Either input image or kernel was %dx%d instead of %dx%d - aborting...\n",
                y,x,height,width);
        }
        SAFE( cudaMallocHost((void **)&_pData, x*y*sizeof(fp)) );
    }
    ~m2dCuda() {
        SAFE( cudaFreeHost(_pData) );
    }
    fp& operator()(int y, int x) { return _pData[y*_width + x]; }
    fp* getLine(int y) { return &_pData[y*_width]; }
};

typedef m2dCuda<KERNEL_SIZE, KERNEL_SIZE> m2dSpecialPSF;
typedef m2dCuda<IMGHEIGHT, IMGWIDTH>      m2dSpecialImage;

#elif defined(USE_EIGEN)

///////////////////////////////////////////////////////////////////////////////
//
// Eigen can be used to implement a fast convolution via SIMD (SSE intrinsics)
//
///////////////////////////////////////////////////////////////////////////////

#include <Eigen/Dense>

// Eigen needs lots of stack space - we will verify that we have enough
// via getrlimit - hence these "getrlimit" dependencies:
#include <sys/time.h>
#include <sys/resource.h>

// Eigen also needs specially aligned buffers because it uses SSE:
// We therefore define the two special types used in the convolution
// (m2dSpecialImage and m2dSpecialPSF) using the Eigen Matrix templates.
typedef Eigen::Matrix<fp, KERNEL_SIZE, KERNEL_SIZE, Eigen::RowMajor> m2dSpecialPSF;
typedef Eigen::Matrix<fp, IMGHEIGHT, IMGWIDTH, Eigen::RowMajor>      m2dSpecialImage;

#else // USE_CUDA_GPU || USE_EIGEN

// Neither Eigen nor CUDA - use the simple m2d typedef 
// from configStraylight.h for the two convolution types.

typedef m2d m2dSpecialPSF;
typedef m2d m2dSpecialImage;

#endif // USE_EIGEN

using namespace std;

//////////////////////////////////////////////////////////////////////////////
// 
// Utility function used for measuring the time taken by each stage
//
//////////////////////////////////////////////////////////////////////////////
void emitStateDuration(int stage)
{
    static long lastTimestamp;

    if (stage == 0) {
        lastTimestamp = getTimeInMS();
        return;
    }
    long currentTimestamp = getTimeInMS();
    debug_printf(
        LVL_INFO,
        "Stage %d took %ld ms\n", stage, currentTimestamp-lastTimestamp);
    lastTimestamp = currentTimestamp;
}

//////////////////////////////////////////////////////////////////////////////
//
// INTERPOL IDL function: C++ implementation for odd-sized arrays
//
//////////////////////////////////////////////////////////////////////////////
void interpol(fp *result, const fp *y, const fp *x, int size, const fp *newX, int newSize)
{
    int i,j,last=size-1,outIdx=0;

    for(j=0; j<newSize; j++) {
        fp X = newX[j];
        // x is in decreasing order, so run it in reverse (increasing)
        for(i=last; i>=0; i--)
            if (x[i]>= X)
                break;
        if (i != -1 && x[i] == X)
            result[outIdx++] = (y[i]);
        else {
            if (i == last) {
                assert(size>1); // at least two needed for a line
                fp deltaX = X-x[last];
                assert(deltaX<0);
                fp stepX = x[last-1] - x[last];
                fp stepY = y[last-1] - y[last];
                result[outIdx++] = (y[last]+stepY*deltaX/stepX);
            } else {
                fp deltaX = X-x[i+1];
                assert(deltaX>0);
                fp stepX = x[i]-x[i+1];
                fp stepY = y[i]-y[i+1];
                result[outIdx++] = (y[i+1] + stepY*deltaX/stepX);
            }
        }
    }
    assert(outIdx == newSize);
}

//////////////////////////////////////////////////////////////////////////////
//
// Logging functions used to dump values in various stages in the computation
// (for debugging and verification of results)
//
//////////////////////////////////////////////////////////////////////////////


// Dictionary of filenames generated so far (used to decide whether
// the stage dump data are opened with 'w' or 'a')
map<string, bool> g_metBefore;

extern int g_bDumpMinimalLevel;

// Function that emits a newline in the specified stage dump file
void dumpNewline(int stageNo, int channelId)
{
    if (stageNo>=g_bDumpMinimalLevel) {
        char stage[128];
        sprintf(stage, "stage%d_%d", stageNo, channelId);
        string filename("output/");
        filename += stage;
        FILE *fp;
        fp = fopen(filename.c_str(), "a");
        if (!fp)
            debug_printf(LVL_PANIC, "Failed to open output file (%s)!\n", stage);
        fputs("\n",fp);
        fclose(fp);
    }
}


// Function that emits a complete matrix (2D) of floats in the specified stage dump file
template <class T>
void dumpFloats(int stageNo, int channelId, int y, int x, T *pData, bool scientific=true)
{
    if (stageNo>=g_bDumpMinimalLevel) {
        char stage[128];
        sprintf(stage, "stage%d_%d", stageNo, channelId);
        string filename("output/");
        filename += stage;
        bool seen = g_metBefore.find(filename) != g_metBefore.end();
        FILE *fp;
        const char *mode = seen?"a":"w";
        fp = fopen(filename.c_str(), mode);
        if (!fp)
            debug_printf(LVL_PANIC, "Failed to open output file (%s)!\n", filename.c_str());
        for(int ii=0; ii<y; ii++) {
            for(int j=0;j<x;j++) {
                fprintf(fp, scientific?" %9.7e":" %12.4f", pData[x*ii+j]);
                if (5==(j%6))  fprintf(fp, "\n");
            }
            fputs("\n",fp);
        }
        fclose(fp);
    }
}


// Function that emits a line of floats in the specified stage dump file
template <class T>
void dumpVector(int stageNo, int channelId, const T *v, int vsize, bool scientific=false)
{
    if (stageNo>=g_bDumpMinimalLevel) {
        char stage[128];
        sprintf(stage, "stage%d_%d", stageNo, channelId);
        string filename("output/");
        filename += stage;
        bool seen = g_metBefore.find(filename) != g_metBefore.end();
        FILE *fp;
        const char *mode = seen?"a":"w";
        fp = fopen(filename.c_str(), mode);
        if (!fp)
            debug_printf(LVL_PANIC, "Failed to open output file (%s)!\n", stage);
        for(int j=0; j<vsize; j++) {
            fprintf(fp, scientific?" %9.7e":" %12.4f", v[j]);
            if (5==(j%6))  fprintf(fp, "\n");
        }
        fputs("\n",fp);
        g_metBefore[filename] = true;
        fclose(fp);
    }
}


//
// The C++ implementation of the StrayLight function, ported as-is from IDL code
//
// Again:
// Comments from the original IDL content start like this: '//#'
// Normal C++ comments are used for porting-related information.
//
// We broke the function in two parts - calculatePSF is the one
// that is called only the first time, and stores its results in
// global variables, so they can be reused in the next invocations.
// (This only happens if USE_PSFCACHE is defined, though)
//
//# NAME:
//# FORWARD_MODEL_SINGLE_PSF_DUAL_RESOLUTION
//#
//# PURPOSE:
//# Given a input scene at the entrance of the instrument, the forward model degrades it by applying straylight following
//# !!!! Add references and
//#
//#
//# CATEGORY:
//# Optics
//#
//# CALLING SEQUENCE:
//# scene_out=forward_model_single_psf_dual_resolution(scene_in, channel, surface_roughness_M1, surface_roughness_M2,surface_roughness_M3, ppm_dust)
//#
//# INPUTS:
//# - scene_in: an array containing the scene radiometry (Irradiances at the entrance of the instrument
//# - channel: [1, 4]
//# - surface_roughness_M1 to M3: surface roughness in meter
//# - ppm_dust: the dust contamination concentration in ppm
//# - pupil_stop: value of the lambertian reflectance contribution of the pupil stop
//# - swir_ghost: =0 => no ghost in the SWIR
//#               =1 => ghost in the SWIR included
//#
//# OPTIONAL INPUT PARAMETERS:
//#
//# OUTPUTS:
//# - an image of the same dimenions than scene_in but degraded by straylight
//#
//# COMMON BLOCKS:
//# None.
//#
//# SIDE EFFECTS:
//# None.
//#
//# RESTRICTIONS:
//# None.
//#
//# PROCEDURE:
//# Model mentioned and described in Taccola in a TN on Proba-V straylight. Largely inspired from Taracola code.
//# PSF model originally developed by Harvey
//# !!! give ref
//#
//#
//# MODIFICATION HISTORY:
//# Written mbouvet@esa.int , 2011 - JULY 25

// Globals that are set once in calculatePSF and reused in subsequent
// invocations of forward_model_single_psf_dual_resolution 
// (that is, if USE_PSFCACHE is activated (which it is, by default)
//
// In moving code outside forward_model_single_psf_dual_resolution and
// into this 'calculatePSF', I am hacking image dimensions to macros
// (for loop unrolling - speed matters!)
#define image_dim_x IMGWIDTH
#define image_dim_y IMGHEIGHT

m1d g_correction_direct_peak(image_dim_x);
int g_extended_image_dim_x = 0;
int g_extended_image_dim_y = 0;
int g_cpt_dim_y = 0;
int g_cpt_dim_x = 0;
int g_lowres_y = 0;
int g_lowres_x = 0;
int g_low_to_high_spatial_resolution = 0;
int g_psf_extent = 0;
m2dSpecialPSF *g_psf_high_res = NULL;
FFTW_PREFIX(complex) *g_fft_psf_low_res = NULL;

void calculatePSF(
    int channel,
    fp surface_roughness_M1,
    fp surface_roughness_M2,
    fp surface_roughness_M3,
    int ppm_dust
    )
{
    // for iterations
    int i,j,k,l;

    //# Focal distance
    fp focal=110e-3;

    //# Size of detectors
    const fp VNIR_detector_size=13e-6; // in meter
    const fp SWIR_detector_size=25e-6; // in meter
    fp detector_size = 0.;

    //# Choose detector size
    if (channel <= 3)
        detector_size = VNIR_detector_size;
    else if (channel == 4)
        detector_size = SWIR_detector_size;

    //# Wavelength
    //
    // (Semantix/ttsiodras: this code was unused, so we commented it out)
    //
    //fp wl[] = {0.45, 0.65, 0.825, 1.6};
    //for (i=0; i<sizeof(wl)/sizeof(wl[0]); i++)
    //  wl[i] *= 1e-6;

    // Derive size of the input image
    //int image_dim_y = input_image._height;
    //int image_dim_x = input_image._width;

    // Make sure we have odd widths - all IDL algorithms have been implemented for odd-sized arrays
    //assert(1 == (image_dim_x&1));

    //# In the case of surface roughness scattering, the peak of the PSF should actualy not be constant across the instantaneous
    //# FOV. It should vary as the square of the cosine of the incidence angle of the chief ray on each mirror
    //# Using the single PSF, we assume an incidence angle of 0 degrees for all points in the instantaneous FOV
    //#
    //# Compute the incidence angle variation across track
    //# We neglect the along track variatons
    //#
    //# From Semen, the values obtained in degrees for the incidence angles in the center and edge of the FOV
    fp i_angles_center_FOV[] = {17.92, 23.03, 11.57};
    fp i_angles_edge_FOV[]   = {32.2,  42.86, 21.23};
    fp delta_incidence_angle[3];
    for(i=0; i<3; i++)
        delta_incidence_angle[i] = i_angles_edge_FOV[i] - i_angles_center_FOV[i];

    //# Computation of the incidence angle variations in radian
    m1d theta0_M1(image_dim_x), theta0_M2(image_dim_x), theta0_M3(image_dim_x);
    for(i=0; i<image_dim_x; i++) {
        theta0_M1[i] = fabs(delta_incidence_angle[0]*M_PI/180.*(i-image_dim_x/2)/(tan(17.3*M_PI/180.)*focal/detector_size))+i_angles_center_FOV[0]*M_PI/180.;
        theta0_M2[i] = fabs(delta_incidence_angle[1]*M_PI/180.*(i-image_dim_x/2)/(tan(17.3*M_PI/180.)*focal/detector_size))+i_angles_center_FOV[1]*M_PI/180.;
        theta0_M3[i] = fabs(delta_incidence_angle[2]*M_PI/180.*(i-image_dim_x/2)/(tan(17.3*M_PI/180.)*focal/detector_size))+i_angles_center_FOV[2]*M_PI/180.;
    }

    emitStateDuration(1);
    dumpVector(1, channel, (fp*) theta0_M1, image_dim_x);
    dumpVector(1, channel, (fp*) theta0_M2, image_dim_x);
    dumpVector(1, channel, (fp*) theta0_M3, image_dim_x);

    //# Computation of TIS
    //#
    //# Define an array where we store the variations of the TIS across track on each mirror are stored
    m2d TIS_surface_roughness_across_track(3,image_dim_x);
    for(i=0;i<3;i++) {
        for(j=0;j<image_dim_x;j++) TIS_surface_roughness_across_track(i,j) = 0.0;
    }

    const int nb_points_TIS_computation=3;
    m1d theta0_M1_downto_3(nb_points_TIS_computation);
    m1d theta0_M2_downto_3(nb_points_TIS_computation);
    m1d theta0_M3_downto_3(nb_points_TIS_computation);
    float ofs=0.0, delta=float(image_dim_x/2)/3.;
    int outputOffset=0, idx=0;
    while(idx<image_dim_x/2) {
        theta0_M1_downto_3[outputOffset  ] = theta0_M1[idx];
        theta0_M2_downto_3[outputOffset  ] = theta0_M2[idx];
        theta0_M3_downto_3[outputOffset++] = theta0_M3[idx];
        ofs += delta;
        idx = int(round(ofs));
    }
    assert(outputOffset == nb_points_TIS_computation);

    //# We subsample the theta0 to compute the TIS which compuatation is time consuming and variations are any quite smooth
    //# with the incidence angle
    if (image_dim_x > 50) {
        //# For each band, we subsample the incidence angle and they we compute the TIS for fewer points
        //# Next we resample to the all the incidence angles

        fp TIS_subsampled[3];
        tis_surface_scattering_harvey(TIS_subsampled, channel, theta0_M1_downto_3, 3, surface_roughness_M1);
        interpol(TIS_surface_roughness_across_track.getLine(0), TIS_subsampled, theta0_M1_downto_3, 3, theta0_M1, image_dim_x);

        tis_surface_scattering_harvey(TIS_subsampled, channel, theta0_M2_downto_3, 3, surface_roughness_M2);
        interpol(TIS_surface_roughness_across_track.getLine(1), TIS_subsampled, theta0_M2_downto_3, 3, theta0_M2, image_dim_x);

        tis_surface_scattering_harvey(TIS_subsampled, channel, theta0_M3_downto_3, 3, surface_roughness_M3);
        interpol(TIS_surface_roughness_across_track.getLine(2), TIS_subsampled, theta0_M3_downto_3, 3, theta0_M3, image_dim_x);
    } else {
        //# We do here the brute force approach: to each incidence angle we compute the TIS
        debug_printf(LVL_PANIC, "Unimplemented, lines must have at least 50 elements (Line %d in %s)\n", __LINE__, __FILE__);
        //#TIS_surface_roughness_across_track[*,0]=tis_surface_scattering_harvey(channel,theta0_M1 , surface_roughness_M1)
        //#TIS_surface_roughness_across_track[*,1]=tis_surface_scattering_harvey(channel,theta0_M2 , surface_roughness_M2)
        //#TIS_surface_roughness_across_track[*,2]=tis_surface_scattering_harvey(channel,theta0_M3 , surface_roughness_M3)
    }

    emitStateDuration(2);
    dumpVector(2, channel, TIS_surface_roughness_across_track.getLine(0), image_dim_x);
    dumpVector(2, channel, TIS_surface_roughness_across_track.getLine(1), image_dim_x);
    dumpVector(2, channel, TIS_surface_roughness_across_track.getLine(2), image_dim_x);

    //#  Define an array where we store thethe TIS in the center of the FOV on each mirror are stored
    fp TIS_surface_roughness_center_FOV[3];
    TIS_surface_roughness_center_FOV[0] = tis_surface_scattering_harvey(channel, i_angles_center_FOV[0]*M_PI/180., surface_roughness_M1);
    TIS_surface_roughness_center_FOV[1] = tis_surface_scattering_harvey(channel, i_angles_center_FOV[1]*M_PI/180., surface_roughness_M2);
    TIS_surface_roughness_center_FOV[2] = tis_surface_scattering_harvey(channel, i_angles_center_FOV[2]*M_PI/180., surface_roughness_M3);

    emitStateDuration(3);
    if (3>=g_bDumpMinimalLevel) {
        char stage[128];
        sprintf(stage, "output/stage3_%d", channel);
        FILE *fp = fopen(stage, "w");
        if (!fp)
            debug_printf(LVL_PANIC, "Failed to open output file!\n");
        for(i=0;i<3;i++)
            fprintf(fp, " %10.8f", TIS_surface_roughness_center_FOV[i]);
        fputs("\n",fp);
    }
    //#
    //#  The correction factor is the ratio of the TIS across track to the TIS in the center of the FOV
    //#  We use the TIS in the center of the FOV to normalise these TIS across track variations because later on in the code
    //#  we make use of Harvey PSF that is computer for the center of the FOV and has the value (1-TIS_center_FOV_M1)(1-TIS_center_FOV_M2)(1-TIS_center_FOV_M3) in its peak
    //#
    for(i=0; i<image_dim_x; i++) {
        g_correction_direct_peak[i] =
            (1. - TIS_surface_roughness_across_track(0,i))*
            (1. - TIS_surface_roughness_across_track(1,i))*
            (1. - TIS_surface_roughness_across_track(2,i));
        g_correction_direct_peak[i] /=
            (1. - TIS_surface_roughness_center_FOV[0])*
            (1. - TIS_surface_roughness_center_FOV[1])*
            (1. - TIS_surface_roughness_center_FOV[2]);
    }

    emitStateDuration(4);
    dumpVector(4, channel, (fp*) g_correction_direct_peak, image_dim_x);

    //#  We split the convolution problem into two convolutions:
    //#  1) A high spatial convolution with the core of the PSF at full spatial resolution:
    //#  this high resolution PSF has a limited spatial extent
    //#  2) A low spatial convolution with the complementatry PSF at a lower spatial resolution
    //#
    //#  Factor defining the ratio between low and high resolution
    g_low_to_high_spatial_resolution=11; //  MUST BE AN ODD NUMBER!!!
    //#
    //#  Extent of the high resolution PSF chosen from looking at the encircled energy plots
    //#  !!! this should be an odd number
    int nb_low_res_bin_in_high_res=1; //  MUST BE AN ODD NUMBER!!!
    assert(1 == (nb_low_res_bin_in_high_res&1));
    g_psf_extent = nb_low_res_bin_in_high_res*g_low_to_high_spatial_resolution; //# this number should be about 500 pixel given the encircled energy of psfs

    if ((g_psf_extent>image_dim_x) || (g_psf_extent>image_dim_y)) {
        debug_printf(LVL_PANIC, "Extent of high res PSF is larger than image itself!");
        fflush(stdout);
        exit(1);
    }

    //#  Define extended input image which is used during the FFT
    //#  This extended image has dimensions which are twice the input_image passed to the routine
    //#  The input_image variable is place at the center of the newly created variable
    //#  The newly created variable must be of dimension at least 3 x dimensions of the original image and should have an odd number of
    //#  columns and lines and should be a multiple of g_low_to_high_spatial_resolution
    //#
    //#  Define starting values of the dimensions of the extended image:
    g_extended_image_dim_x = 3*image_dim_x;
    g_extended_image_dim_y = 3*image_dim_y;
    //#
    //#  Iterate on dimensions that are odd and multiple of g_low_to_high_spatial_resolution and larger than 3*image_dim_x
    //#
    g_cpt_dim_x=0;
    do {
        g_cpt_dim_x++;
    } while (!(((g_extended_image_dim_x+g_cpt_dim_x)%2 == 1) && ((g_extended_image_dim_x+g_cpt_dim_x)%g_low_to_high_spatial_resolution == 0)));
    g_extended_image_dim_x = g_extended_image_dim_x+g_cpt_dim_x;

    g_cpt_dim_y = 0;
    do {
        g_cpt_dim_y++;
    } while (!(((g_extended_image_dim_y+g_cpt_dim_y)%2 == 1) && ((g_extended_image_dim_y+g_cpt_dim_y)%g_low_to_high_spatial_resolution == 0)));
    g_extended_image_dim_y = g_extended_image_dim_y+g_cpt_dim_y;
    debug_printf(LVL_INFO, "Input image resolution:    %dx%d\n", image_dim_y, image_dim_x);
    debug_printf(LVL_INFO, "Extended image resolution: %dx%d\n", g_extended_image_dim_y, g_extended_image_dim_x);

    //# ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    //#  Compute the PSF
    //# ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    //#
    //#  First create an array with of size of the image which contains the distance to the Gaussian image in the focal
    //#  plane in micrometer
    //# pix_coordinate=fltarr(2*image_dim_x-1, 2* image_dim_y-1,2)
    //# pix_coordinate[*,*,0]=indgen(2* image_dim_x-1)#transpose(intarr(2* image_dim_y-1)+1)
    //# pix_coordinate[*,*,1]=transpose(indgen(2*image_dim_y-1)#transpose((intarr(2*image_dim_x-1)+1)))
    //# radius=sqrt((pix_coordinate[*,*,0]-image_dim_x+1)^2+(pix_coordinate[*,*,1]-image_dim_y+1)^2)

    m2d *pix_coordinate[2];
    pix_coordinate[0] = new m2d(g_extended_image_dim_y, g_extended_image_dim_x);
    pix_coordinate[1] = new m2d(g_extended_image_dim_y, g_extended_image_dim_x);
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<g_extended_image_dim_y; i++) {
        for(int j=0; j<g_extended_image_dim_x; j++) {
            pix_coordinate[0]->operator()(i,j) = j;
            pix_coordinate[1]->operator()(i,j) = i;
        }
    }

    emitStateDuration(6);
    for(int i=0; i<g_extended_image_dim_y; i++)
        dumpVector(6, channel, pix_coordinate[0]->getLine(i), g_extended_image_dim_x);
    dumpNewline(6, channel);
    for(int i=0; i<g_extended_image_dim_y; i++)
        dumpVector(6, channel, pix_coordinate[1]->getLine(i), g_extended_image_dim_x);

    m2d radius(g_extended_image_dim_y, g_extended_image_dim_x);
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<g_extended_image_dim_y; i++) {
        for(int j=0; j<g_extended_image_dim_x; j++) {
            //#  Multiply by the detector size
            radius(i,j) = detector_size*sqrt(pow((*pix_coordinate[0])(i,j)-g_extended_image_dim_x/2,2)+pow((*pix_coordinate[1])(i,j)-g_extended_image_dim_y/2,2));
        }
        //dumpVector(66, radius[i]);
    }

    //#  Compute the PSF from the 3 mirrors mirror due to the surface roughness
    m2d psf_mirror_harvey(g_extended_image_dim_y, g_extended_image_dim_x);
    harvey_psf(psf_mirror_harvey, radius, channel,
        surface_roughness_M1, surface_roughness_M2 , surface_roughness_M3, true);

    emitStateDuration(7);
    for(int i=0; i<g_extended_image_dim_y; i++)
        dumpVector(7, channel, psf_mirror_harvey.getLine(i), g_extended_image_dim_x, true);

    //#  Compute the Mie scattering due to particulate contamination psf for the 3 mirrors
    m2d psf_mirror_dust(g_extended_image_dim_y, g_extended_image_dim_x);
    if (ppm_dust > 0) {
        particulate_contamination_harvey_psf(psf_mirror_dust, radius, channel);
    } else {
        assert(psf_mirror_harvey._height == g_extended_image_dim_y);
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
        for(int i=0; i<g_extended_image_dim_y; i++) {
            for(int j=0; j<g_extended_image_dim_x; j++) {
                psf_mirror_dust(i,j) = (radius(i,j) == 0.)?1.:0.;
            }
        }
    }

    assert(psf_mirror_harvey._height == psf_mirror_dust._height);
    assert(psf_mirror_harvey._width == psf_mirror_dust._width);

    m2d psf_totale(g_extended_image_dim_y, g_extended_image_dim_x);
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<g_extended_image_dim_y; i++) {
        for(int j=0; j<g_extended_image_dim_x; j++) {
            if (radius(i,j) == 0.)
                //# Multiply the two PSF direct components
                psf_totale(i,j) = psf_mirror_harvey(i,j) * psf_mirror_dust(i,j);
            else
                //# Add the two PSF diffuse components
                psf_totale(i,j) = psf_mirror_harvey(i,j) + psf_mirror_dust(i,j);
        }
    }

    emitStateDuration(8);
    for(int i=0; i<g_extended_image_dim_y; i++)
        dumpVector(8, channel, psf_totale.getLine(i), g_extended_image_dim_x, true);

    //#  Define the high res psf
#ifndef USE_PSFCACHE
    // if caching is not desired, reset each time
    if (!g_psf_high_res) {
        delete g_psf_high_res;
        g_psf_high_res = NULL;
    }
#endif
    if (!g_psf_high_res)
        g_psf_high_res = new m2dSpecialPSF(g_psf_extent, g_psf_extent);
    for(i=0; i<g_psf_extent; i++) {
        for(j=0; j<g_psf_extent; j++) {
            (*g_psf_high_res)(i,j) = psf_totale
                ((g_extended_image_dim_y-g_psf_extent)/2 + i,
                 (g_extended_image_dim_x-g_psf_extent)/2 + j);
        }
    }

    //#  Define the low res psf
    //#  Remove the high res part of the psf
    //
    //  ttsiodras: avoid copying to new table by overwriting psf_totale - which is not reused below)
    for(i=0; i<g_psf_extent; i++) {
        for(j=0; j<g_psf_extent; j++) {
            psf_totale
                ((g_extended_image_dim_y-g_psf_extent)/2 + i,
                 (g_extended_image_dim_x-g_psf_extent)/2 + j) = 0.;
        }
    }
    //#  Resize to low res
    g_lowres_y = g_extended_image_dim_y/g_low_to_high_spatial_resolution;
    g_lowres_x = g_extended_image_dim_x/g_low_to_high_spatial_resolution;
    FFTWRealType *psf_low_res = (FFTWRealType*) FFTW_PREFIX(malloc)(sizeof(FFTWRealType) * g_lowres_y * g_lowres_x);
    memset(psf_low_res, 0, sizeof(FFTWRealType) * g_lowres_y * g_lowres_x);

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for(i=0; i<g_extended_image_dim_y; i+=g_low_to_high_spatial_resolution) {
        for(j=0; j<g_extended_image_dim_x; j+=g_low_to_high_spatial_resolution) {
            fp total=0.;
            for(k=0; k<g_low_to_high_spatial_resolution; k++)
                for(l=0; l<g_low_to_high_spatial_resolution; l++)
                    total += psf_totale(i+k,j+l);
            psf_low_res[
                    g_lowres_x*(i/g_low_to_high_spatial_resolution) +
                             (j/g_low_to_high_spatial_resolution)]
                = total/(g_low_to_high_spatial_resolution*g_low_to_high_spatial_resolution);
        }
    }

    //for(int ii=0; ii<g_psf_extent; ii++)
    //    dumpVector(92, channel, &(*g_psf_high_res)(ii, 0), g_psf_extent, true);

    emitStateDuration(9);
    dumpFloats(9, channel, g_lowres_y, g_lowres_x, psf_low_res);

    // This following section was further below - I moved it here, since psf_low_res
    // is computed only the first time - i.e. it is skipped over via the goto
    // in subsequent images...

    //#   Compute the FFT of the PSF
    //# fft_psf_low_res=fft(psf_low_res)
    //#
#ifndef USE_PSFCACHE
    // if caching is not desired, reset each time
    if (!g_fft_psf_low_res) {
        FFTW_PREFIX(free)(g_fft_psf_low_res);
        g_fft_psf_low_res = NULL;
    }
#endif
    if (!g_fft_psf_low_res)
        g_fft_psf_low_res = (FFTW_PREFIX(complex)*) FFTW_PREFIX(malloc)(sizeof(FFTW_PREFIX(complex)) * g_lowres_x * g_lowres_y);
    FFTW_PREFIX(plan) psf_low_res_plan =
        FFTW_PREFIX(plan_dft_r2c_2d)(g_lowres_y, g_lowres_x, psf_low_res, g_fft_psf_low_res, FFTW_ESTIMATE);
    FFTW_PREFIX(execute)(psf_low_res_plan);
    FFTW_PREFIX(destroy_plan)(psf_low_res_plan);

    emitStateDuration(91);

}

#ifdef USE_CUDA_GPU
extern void cudaConvolution(
    fp *cudaMainMatrix, fp *cudaMainKernel, fp *cudaOutputImage/*, int inX, int inY, int kX, int kY*/);
#endif // USE_CUDA_GPU

void forward_model_single_psf_dual_resolution(
    m2d& output_image,
    m2d& input_image,
    int channel,
    fp surface_roughness_M1,
    fp surface_roughness_M2,
    fp surface_roughness_M3,
    int ppm_dust,
    fp pupil_stop,
    bool /* swir_ghost */,   // unused for now
    bool logStages)
{
    // for iterations
    int i,j,k,l;

    // !!! OBSOLETE - we expect image size to remain constant
    // !!! (loop unrolling!)
    // Derive size of the input image 
    //int image_dim_y = input_image._height;
    //int image_dim_x = input_image._width;

    emitStateDuration(0);

#ifdef USE_PSFCACHE
    // static flag used to avoid re-doing the PSF calculation upon each invocation
    static int oldChannel = -1;

    if (oldChannel != channel) {
        oldChannel = channel;
#endif // USE_PSFCACHE

        calculatePSF(
            channel,
            surface_roughness_M1,
            surface_roughness_M2,
            surface_roughness_M3,
            ppm_dust
        );
#ifdef USE_PSFCACHE
    } else {
        debug_printf(LVL_INFO, "Re-using PSF cache...\n");
    }

    if (image_dim_y != input_image._height || image_dim_x != input_image._width)
        debug_printf(
            LVL_PANIC,
            "forward_model_single_psf_dual_resolution violated the contract!\n"
            "It was called with input image size %dx%d instead of %dx%d - aborting...\n",
            input_image._width, input_image._height, IMGWIDTH, IMGHEIGHT);
#endif // USE_PSFCACHE

    //#  Apply ratio of normal to variable with incidence angle TIS to the input image
    static m2dSpecialImage input_image_corrected(image_dim_y, image_dim_x);
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for(int i_y=0; i_y<image_dim_y; i_y++) {
        for(int j=0; j<image_dim_x; j++)
           input_image_corrected(i_y,j) = input_image(i_y,j)*g_correction_direct_peak[j];
    }

    //emitStateDuration(91);
    //for(int i_y=0; i_y<image_dim_y; i_y++)
    //    dumpVector(91, channel, input_image_corrected.getLine(i_y), image_dim_x);

    static m2d input_extended_image(g_extended_image_dim_y, g_extended_image_dim_x);

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for(i=image_dim_y+g_cpt_dim_y/2; i<=2*image_dim_y-1+g_cpt_dim_y/2; i++)
        for(int j=image_dim_x+g_cpt_dim_x/2; j<=2*image_dim_x-1+g_cpt_dim_x/2; j++)
            input_extended_image(i,j) = input_image_corrected(i-image_dim_y-g_cpt_dim_y/2,j-image_dim_x-g_cpt_dim_x/2);

    //emitStateDuration(92);
    //for(int i=0; i<g_extended_image_dim_y; i++)
    //    dumpVector(55, channel, input_extended_image.getLine(i), g_extended_image_dim_x);

    //# ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    //#  The convolution of the PSF with the image
    //# ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    //#
    //#  We split the convolution problem into two convolutions:
    //#  1) A high spatial convolution with the core of the PSF at full spatial resolution:
    //#  this high resolution PSF has a limited spatial extent
    //#  2) A low spatial convolution with the complementatry PSF at a lower spatial resolution
    //#
    //#  Compute the low res input image
    FFTWRealType *low_res_input_extended_image =
        (FFTWRealType*) FFTW_PREFIX(malloc)(sizeof(FFTWRealType) * g_lowres_y * g_lowres_x);
    //debug_printf(LVL_INFO, "Low resolution image:      %dx%d\n", g_lowres_y, g_lowres_x);
    memset(low_res_input_extended_image, 0, sizeof(FFTWRealType) * g_lowres_y * g_lowres_x);

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for(i=0; i<g_extended_image_dim_y; i+=g_low_to_high_spatial_resolution) {
        for(j=0; j<g_extended_image_dim_x; j+=g_low_to_high_spatial_resolution) {
            FFTWRealType total=0.;
            for(k=0; k<g_low_to_high_spatial_resolution; k++)
                for(l=0; l<g_low_to_high_spatial_resolution; l++)
                    total += input_extended_image(i+k,j+l);
            low_res_input_extended_image[
                    g_lowres_x*(i/g_low_to_high_spatial_resolution) +
                    j/g_low_to_high_spatial_resolution] =
                total/(g_low_to_high_spatial_resolution*g_low_to_high_spatial_resolution);
        }
    }

    //dumpFloats(85, channel, g_lowres_y, g_lowres_x, low_res_input_extended_image, false);
    //emitStateDuration(93);

    //# ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    //#  Compute the high res part of the output via convolution
    //# ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    //#  Convolution of the high res psf with the high repeating the values at the edge of the image
    //#  Here an important assumption is that there is not signal outside the strict field of view of the input image

    int inY = image_dim_y;
    int inX = image_dim_x;
    if (inX != IMGWIDTH || inY != IMGHEIGHT)
        debug_printf(
            LVL_PANIC,
            "forward_model_single_psf_dual_resolution violated the contract!\n"
            "It ends up with an input image of %dx%d instead of %dx%d! Aborting...\n",
            inX, inY, IMGWIDTH, IMGHEIGHT);

    // Using constants instead of kX and kY causes tremendous speed difference - due to loop unrolling!
    int kY = g_psf_extent;
    int kX = g_psf_extent;
    if (kX != KERNEL_SIZE || kY != KERNEL_SIZE)
        debug_printf(
            LVL_PANIC,
            "forward_model_single_psf_dual_resolution violated the contract!\n"
            "It ends up with a pfs_high_res of %dx%d instead of %dx%d! Aborting...\n",
            kX, kY, KERNEL_SIZE, KERNEL_SIZE);
    // Ugly hack - but I may need to revert to normal vars in the future, so this will do.
    #define kX KERNEL_SIZE
    #define kY KERNEL_SIZE

#ifdef USE_CUDA_GPU
    m2d high_res_output_image(inY, inX);
    fp *cudaMainKernel=NULL;
    SAFE( cudaMalloc((void**)&cudaMainKernel, kX*kY*sizeof(fp)) );
    SAFE( cudaMemcpy(cudaMainKernel, g_psf_high_res->_pData, kX*kY*sizeof(fp), cudaMemcpyHostToDevice) );

    fp *cudaMainMatrix=NULL;
    fp *cudaOutputImage=NULL;
    SAFE( cudaMalloc((void**)&cudaMainMatrix, inX*inY*sizeof(fp)) );
    SAFE( cudaMalloc((void**)&cudaOutputImage, inX*inY*sizeof(fp)) );

    SAFE( cudaMemcpy(cudaMainMatrix, input_image_corrected._pData, inX*inY*sizeof(fp), cudaMemcpyHostToDevice) );
    cudaConvolution(cudaMainMatrix, cudaMainKernel, cudaOutputImage/*, inX, inY, kX, kY*/);
    SAFE( cudaMemcpy(high_res_output_image._pData, cudaOutputImage, inX*inY*sizeof(fp), cudaMemcpyDeviceToHost) );

    SAFE( cudaFree(cudaOutputImage) );
    SAFE( cudaFree(cudaMainKernel) );
    SAFE( cudaFree(cudaMainMatrix) );

#elif defined(USE_EIGEN)
    using namespace Eigen;

    Matrix<fp, IMGHEIGHT, IMGWIDTH, RowMajor> high_res_output_image;

    high_res_output_image.setZero();
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (i=0; i<IMGHEIGHT-KERNEL_SIZE; i++) {
        fp *high_res_output_image_line = &high_res_output_image(i+KERNEL_SIZE/2,KERNEL_SIZE/2);
        for (j=0; j<IMGWIDTH-KERNEL_SIZE; j++) {
            fp sum = input_image_corrected.block<KERNEL_SIZE,KERNEL_SIZE>(i,j).cwiseProduct(*g_psf_high_res).sum();
            //high_res_output_image((i+KERNEL_SIZE/2)%IMGHEIGHT, (j+KERNEL_SIZE/2)%IMGWIDTH) = sum;
            *high_res_output_image_line++ = sum;
        }
        for (j=IMGWIDTH-KERNEL_SIZE; j<IMGWIDTH; j++) {
            fp sum = 0;
            for(k=0; k<KERNEL_SIZE; k++) {
                for(l=0; l<KERNEL_SIZE; l++) {
                    sum += input_image_corrected((i+k)%IMGHEIGHT, (j+l)%IMGWIDTH)*(*g_psf_high_res)(k,l);
                }
            }
            high_res_output_image((i+KERNEL_SIZE/2)%IMGHEIGHT, (j+KERNEL_SIZE/2)%IMGWIDTH) = sum;
        }
    }
    for (i=IMGHEIGHT-KERNEL_SIZE; i<IMGHEIGHT; i++) {
        for (j=0; j<IMGWIDTH; j++) {
            fp sum = 0;
            for(k=0; k<KERNEL_SIZE; k++) {
                for(l=0; l<KERNEL_SIZE; l++) {
                    sum += input_image_corrected((i+k)%IMGHEIGHT, (j+l)%IMGWIDTH)*(*g_psf_high_res)(k,l);
                }
            }
            high_res_output_image((i+KERNEL_SIZE/2)%IMGHEIGHT, (j+KERNEL_SIZE/2)%IMGWIDTH) = sum;
        }
    }

#else // Neither Eigen nor CUDA - plain (potentially OpenMP) code.

    m2d high_res_output_image(inY, inX);

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (i=0; i<inY; i++) {
        int sY = i+kY/2;
        if (sY >= inY) sY -= inY;
        fp *high_res_output_image_line = high_res_output_image.getLine(sY);
        for (j=0; j<inX; j++) {
            fp sum=0.;
            for(k=0; k<kY; k++) {
                int ofsY = i+k;
                if (ofsY>=inY) ofsY -= inY;
                fp *input_image_corrected_line = input_image_corrected.getLine(ofsY);
                fp *psf_high_res_line = g_psf_high_res->getLine(k);
                for(l=0; l<kX; l++) {
                    int ofsX = j+l;
                    //if (ofsX>=inX) ofsX -= inX;
                    if (__builtin_expect(ofsX>=inX,0)) ofsX -= inX;
                    sum += input_image_corrected_line[ofsX]*psf_high_res_line[l];
                }
            }
            int outOfsX = j+kX/2;
            if (outOfsX>=inX) outOfsX -= inX;
            high_res_output_image_line[outOfsX] = sum;
        }
    }

#endif // USE_CUDA_GPU

    emitStateDuration(10);
    for (i=0; i<inY; i++)
#ifndef USE_EIGEN
        dumpVector(10, channel, high_res_output_image.getLine(i), high_res_output_image._width);
#else
        dumpVector(10, channel, &high_res_output_image(i, 0), IMGWIDTH);
#endif

    //#  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    //#   Compute the low res part of the output via FFT
    //#  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    //#   Compute the FFT of the extended image
    //# fft_image_low_res=fft(low_res_input_extended_image)
    //#
    //#   Output should be the inverse FFT of the product of the FFTs but...
    //# low_res_extended_output_image=fft(fft_psf_low_res*fft_image_low_res, /inverse)

    FFTW_PREFIX(complex) *fft_image_low_res = (FFTW_PREFIX(complex)*) FFTW_PREFIX(malloc)(sizeof(FFTW_PREFIX(complex)) * g_lowres_x * g_lowres_y);
    FFTW_PREFIX(plan) low_res_input_extended_image_plan =
        FFTW_PREFIX(plan_dft_r2c_2d)(g_lowres_y, g_lowres_x, low_res_input_extended_image, fft_image_low_res, FFTW_ESTIMATE);
    FFTW_PREFIX(execute)(low_res_input_extended_image_plan);

    int span = (1+g_lowres_x)/2;
    FFTW_PREFIX(complex) *revfft_psf_low_res = (FFTW_PREFIX(complex)*) FFTW_PREFIX(malloc)(sizeof(FFTW_PREFIX(complex)) * g_lowres_x * g_lowres_y);
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<g_lowres_y; i++)
        for(int j=0; j<span; j++) {
            fp re1 = g_fft_psf_low_res[i*span + j][0];
            fp im1 = g_fft_psf_low_res[i*span + j][1];
            fp re2 = fft_image_low_res[i*span + j][0];
            fp im2 = fft_image_low_res[i*span + j][1];
            revfft_psf_low_res[i*span + j][0] = re1*re2-im1*im2;
            revfft_psf_low_res[i*span + j][1] = re1*im2+re2*im1;
            revfft_psf_low_res[i*span + j][0] /= g_lowres_x*g_lowres_y;
            revfft_psf_low_res[i*span + j][1] /= g_lowres_x*g_lowres_y;
            revfft_psf_low_res[i*span + j][0] /= g_lowres_x*g_lowres_y;
            revfft_psf_low_res[i*span + j][1] /= g_lowres_x*g_lowres_y;
        }
    FFTWRealType *low_res_extended_output_image = (FFTWRealType*) FFTW_PREFIX(malloc)(sizeof(FFTWRealType) * g_lowres_y * g_lowres_x);
    FFTW_PREFIX(plan) low_res_extended_output_image_plan =
        FFTW_PREFIX(plan_dft_c2r_2d)(g_lowres_y, g_lowres_x, revfft_psf_low_res, low_res_extended_output_image, FFTW_ESTIMATE);
    FFTW_PREFIX(execute)(low_res_extended_output_image_plan);
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<g_lowres_y*g_lowres_x; i++) {
        // Thanassis, stop hacking stuff:
        //*p++ &= 0x7FFFFFFF; // fast single-precision absolute value
        low_res_extended_output_image[i] = fabs( low_res_extended_output_image[i]) * (g_lowres_y*g_lowres_x);
    }

    emitStateDuration(11);
    dumpFloats(11, channel, g_lowres_y, g_lowres_x, low_res_extended_output_image, false);

    FFTW_PREFIX(destroy_plan)(low_res_extended_output_image_plan);
    FFTW_PREFIX(destroy_plan)(low_res_input_extended_image_plan);
    FFTW_PREFIX(free)(fft_image_low_res);
    FFTW_PREFIX(free)(low_res_input_extended_image);
    FFTW_PREFIX(free)(revfft_psf_low_res);
    //Don't free these anymore, they are static - re-used in subsequent invocations of this function!
    //FFTW_PREFIX(free)(fft_psf_low_res);
    //FFTW_PREFIX(free)(psf_low_res);

    //# low_res_extended_output_image= shift(low_res_extended_output_image,g_extended_image_dim_x/2/g_low_to_high_spatial_resolution+1,g_extended_image_dim_y/2/g_low_to_high_spatial_resolution+1 )

    m2d low_res_extended_output_image_shifted(g_lowres_y,g_lowres_x);
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<g_lowres_y; i++) {
        for(int j=0; j<g_lowres_x; j++) {
            unsigned in = (i+g_extended_image_dim_y/2/g_low_to_high_spatial_resolution+1)%g_lowres_y;
            unsigned jn = (j+g_extended_image_dim_x/2/g_low_to_high_spatial_resolution+1)%g_lowres_x;
            low_res_extended_output_image_shifted(in,jn) = low_res_extended_output_image[g_lowres_x*i+j];
        }
    }
    FFTW_PREFIX(free)(low_res_extended_output_image);  // no longer needed

    //emitStateDuration(122);
    //dumpFloats(122, channel, g_lowres_y, g_lowres_x, low_res_extended_output_image_shifted._pData);

    //#  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    //#   Regridding of the low res output to a high res output size
    //#  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    //#
    //#   Rescaling of the low resolution image is done here
    //#   There is an important option that we use: the keyword /interp
    //#   Using no keyword, would imply a rescalling of the low res image by nearest neighbourg which has the disadvantage to
    //#   create large square of g_low_to_high_spatial_resolution x g_low_to_high_spatial_resolution pixels in the final output
    //#   By using the keyword /interp we work around this issue but create others in the vicinity of highly contrasted areas
    //# low_res_extended_output_image=g_low_to_high_spatial_resolution^2*congrid(low_res_extended_output_image, g_extended_image_dim_x, g_extended_image_dim_y, /interp, /center)

    // This port of congrid works only for odd input and output sizes
    assert(1==(g_extended_image_dim_y&1));
    assert(1==(g_extended_image_dim_x&1));
    assert(1==(g_lowres_x&1));
    assert(1==(g_lowres_y&1));

    int ileny = g_lowres_y;
    int oleny = g_extended_image_dim_y;
    double stepy = ileny;
    stepy /= oleny;

    int ilenx = g_lowres_x;
    int olenx = g_extended_image_dim_x;
    double stepx = ilenx;
    stepx /= olenx;

    double startIdxY = ileny/2;
    startIdxY -= stepy*(oleny/2);
    m2d low_res_extended_output_image_linterp(g_extended_image_dim_y, g_extended_image_dim_x);
    m2d& inp = low_res_extended_output_image_shifted;
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    //for(int i=0; i<g_extended_image_dim_y; i++) {
    for(int i=g_cpt_dim_y/2 + image_dim_y; i<g_cpt_dim_y/2 + 2*image_dim_y; i++) {

        startIdxY = ileny/2 - stepy*(oleny/2) + i*stepy;

        double dOfsY;
        int stIdxY = int(startIdxY);
        if (startIdxY<0) {
            stIdxY=0;
            dOfsY = 0.;
        } else if (stIdxY>ileny-2) {
            stIdxY=ileny-2;
            dOfsY = 1.;
        } else
            dOfsY = startIdxY - stIdxY;

        double startIdxX = ilenx/2;
        startIdxX -= stepx*(olenx/2);
        for(int j=0; j<g_extended_image_dim_x; j++) {
            //
            // Obvious optimization: the code below only reads from 
            // a central square inside low_res_extended_output_image_linterp:
            //
            //  for(int i=0; i<image_dim_y; i++) {
            //      for(int j=0; j<image_dim_x; j++) {
            //
            //      ... read from low_res_extended_output_image_linterp(
            //          i+image_dim_y+g_cpt_dim_y/2,
            //          j+image_dim_x+g_cpt_dim_x/2)
            //
            // ... so naturally, we skip over interpolating for these areas!
            if (j<g_cpt_dim_x/2 + image_dim_x) {
                startIdxX += stepx;
                continue;
            }
            if (j>g_cpt_dim_x/2 + 2*image_dim_x)
                break;

            double dOfsX;
            int stIdxX = int(startIdxX);
            if (startIdxX<0) {
                stIdxX=0;
                dOfsX = 0.;
            } else if (stIdxX>ilenx-2) {
                stIdxX=ilenx-2;
                dOfsX = 1.;
            } else
                dOfsX = startIdxX - stIdxX;

            double tl = inp(stIdxY,   stIdxX);
            double bl = inp(stIdxY+1, stIdxX);
            double tr = inp(stIdxY,   stIdxX+1);
            double br = inp(stIdxY+1, stIdxX+1);

            double lColumn = tl + dOfsY*(bl-tl);
            double rColumn = tr + dOfsY*(br-tr);
            low_res_extended_output_image_linterp(i,j) =
                g_low_to_high_spatial_resolution*g_low_to_high_spatial_resolution*
                (lColumn + (rColumn-lColumn)*dOfsX);

            startIdxX += stepx;
        }
    }


    emitStateDuration(12);
    dumpFloats(12, channel, g_extended_image_dim_y, g_extended_image_dim_x, low_res_extended_output_image_linterp._pData, false);

    //#   Add the two componemts
    //# low_res_output_image=low_res_extended_output_image[image_dim_x+g_cpt_dim_x/2:2*image_dim_x-1+g_cpt_dim_x/2,image_dim_y+g_cpt_dim_y/2:2*image_dim_y-1+g_cpt_dim_y/2]
    //# output_image=low_res_output_image+high_res_output_image

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<image_dim_y; i++) {
        for(int j=0; j<image_dim_x; j++) {
            output_image(i,j) =
                high_res_output_image(i,j) +
                low_res_extended_output_image_linterp(
                    i+image_dim_y+g_cpt_dim_y/2,
                    j+image_dim_x+g_cpt_dim_x/2);
        }
    }

    //emitStateDuration(125);
    //dumpFloats(125, channel, image_dim_y, image_dim_x, output_image._pData, false);

    //#  Add the pupil stop contribution
    //#  output_image=output_image+total(input_image)*pupil_stop
    double total=0.;
#ifdef USE_OPENMP
#pragma omp parallel for reduction(+:total)
#endif
    for(int i_y=0; i_y<image_dim_y; i_y++)
        for(int j=0; j<image_dim_x; j++)
           total += input_image(i_y,j);

    total*=pupil_stop;
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<image_dim_y; i++)
        for(int j=0; j<image_dim_x; j++)
            output_image(i,j) += total;

    emitStateDuration(13);
    if (logStages)
        dumpFloats(13, channel, image_dim_y, image_dim_x, output_image._pData, false);

    if (channel == 4) {
        debug_printf(LVL_PANIC, "Not implemented yet - not used in ESA's prototype. (Line %d in %s)\n", __LINE__, __FILE__);
        //#  Retrieve the PSF of the ghost in the SWIR
        //#if channel eq 4 then begin
        //#  if swir_ghost eq 1 then begin
        //#    ; Read the 160 x 5 pixels SWIR ghost
        //#    psf_swir_ghost=read_psf_ghost_swir()
        //#
        //#    size_swir_ghost=size(psf_swir_ghost)
        //#
        //#    ; Crop the psf if the image is smaller that the psf
        //#    if (image_dim_x lt size_swir_ghost[1]) then begin
        //#      psf_swir_ghost=psf_swir_ghost[size_swir_ghost[1]/2-image_dim_x/2+1:size_swir_ghost[1]/2+image_dim_x/2-1, *]
        //#    endif
        //#
        //#    if (image_dim_y lt size_swir_ghost[2]) then begin
        //#      psf_swir_ghost=psf_swir_ghost[*, size_swir_ghost[2]/2-image_dim_y/2+1:size_swir_ghost[2]/2+image_dim_y/2-1]
        //#    endif
        //#    ;output_image=convol(output_image, psf_swir_ghost, 1, /edge_zero)
        //#    output_image=convol(output_image, psf_swir_ghost, 1, /edge_wrap)
        //#  endif
        //#endif
    }
}
