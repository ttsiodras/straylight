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

#include <string>
#include <stdlib.h>

#include <math.h>

#include <getopt.h>

#include "configStraylight.h"
#include "utilities.h"
#include "forward_model_single_psf_dual_resolution.h"

#ifdef USE_EIGEN
// Eigen needs lots of stack space - we will verify it's enough
// via getrlimit - hence the getrlimit dependencies...
#include <sys/time.h>
#include <sys/resource.h>
#endif // USE_EIGEN

using namespace std;

int g_bDumpMinimalLevel = 13;

void usage(char *argv[])
{
#ifndef PACKAGE_STRING
#define PACKAGE_STRING "StrayLight 1.x"
#endif
    cerr << PACKAGE_STRING;
#ifdef __TIMESTAMP__
    cerr << " (" << __TIMESTAMP__ << ")";
#endif

    cerr << "\n\nCompiled with:\n\n";
#ifdef USE_DOUBLEPRECISION
    cerr << "- Double precision floating point\n";
#else
    cerr << "- Single precision floating point\n";
#endif
#ifdef USE_CUDA_GPU
    cerr << "- CUDA-based convolution\n";
#endif
#ifdef USE_EIGEN
    cerr << "- Eigen-based convolution\n";
#endif
#ifdef USE_OPENMP
    cerr << "- OpenMP-based multithreading\n";
#endif
#ifdef USE_PSFCACHE
    cerr << "- Automatic re-use of calculated PSF data for contiguous\n  calls to identical channels\n";
#endif
    cerr << "\nUsage: " << basename(argv[0]) << " [OPTIONS]\n\n";
    cerr << "  -h            this help\n";
    cerr << "  -v            increase verbosity\n";
    cerr << "  -V            show version\n";
    cerr << "  -b            run benchmark (50 images)\n";
    cerr << "  -i filename   instead of the ESA test image, process this file\n";
    cerr << "  -c channel    process this channel\n";
    cerr << "  -d N          dump log files from computation stages >= N\n";
    cerr << "                (default: 13, i.e. the final result is saved,\n";
    cerr << "                 and never used during benchmarking (no output).\n";
    exit(0);
}


int main(int argc, char *argv[])
{
    int iVerbosityLevel = 1;
    bool bBenchmark = false;
    string sInputFilename = "";
    int iChannel = 1;

    int c;
    while ((c = getopt (argc, argv, "hvVbi:c:d:")) != -1) {
	switch(c) {
	case 'h':
	    usage(argv);
	    break;
	case 'v':
	    iVerbosityLevel++;
	    break;
	case 'V':
            cout << "Version " << PACKAGE_VERSION << endl;
            exit(0);
        case 'b':
	    bBenchmark = true;
	    break;
        case 'i':
            sInputFilename = (const char *)optarg;
	    break;
        case 'c':
	    iChannel = atoi(optarg);
	    break;
        case 'd':
	    g_bDumpMinimalLevel = atoi(optarg);
	    break;
        default:
            usage(argv);
            break;
        }
    }

    if (bBenchmark)
        g_bDumpMinimalLevel = 1000; // Never log files when benchmarking

#ifdef USE_EIGEN
    // Eigen needs lots of stack space
    struct rlimit rlim;
    if (getrlimit(RLIMIT_STACK, &rlim)) {
        puts("You've chosen to use Eigen - checking stack space via getrlimit...");
        puts("Failed to invoke getrlimit!");
        fflush(stdout);
        exit(1);
    } else if (rlim.rlim_cur < 73728*1024) {
        puts("You've chosen to use Eigen - checking stack space via getrlimit...");
        puts("Please invoke 'ulimit -s 73728' and run again - Eigen needs lots of stack.");
        fflush(stdout);
        exit(1);
    }
#endif // USE_EIGEN

    iVerbosityLevel = min(iVerbosityLevel, (int)LVL_INFO); // upper bound
    g_debugLevel = DebugLevel(iVerbosityLevel);

    //# Define the surface roughness and contamination
    fp surf_rough_M1=4e-9;
    fp surf_rough_M2=2e-9;
    fp surf_rough_M3=4e-9;
    fp dust_ppm=1000.;

    //# pupil stop is defined later on in loop

    //# SWIR ghost
    int swir_ghost=1;

    //# Define the reference levels
    //
    // (Semantix/ttsiodras: unused, commented)
    //
    //const fp L4_radiance[] = {567, 446,296,58};
    //const fp L3_radiance[] = {236, 231,212,38};
    //const fp L2_radiance[] = {111, 110,106,20};
    //const fp L1_radiance[] = {39, 10,4,0.6};
    //const fp L1_radiance_new[] = {42,15,27,1.7};
    //const fp L3_radiance_new[] = {195, 252, 199, 45};

    //# Define the NEdR in W.m-2.sr-1.micrometer-1
    //
    // (Semantix/ttsiodras: unused, commented)
    //
    //const fp nedr[] = {0.59, 0.33, 0.27, 0.06};

    const fp focal=110e-3;

    //# Field of view of the instrument in degrees
    const fp alpha_AC_max=17.3; // 0.25*2*0.6771;17.3 ; Max across track viewing angle in degrees
    const fp alpha_AL_max=2.75; // 0.25*2*0.6771;2.75 ; Max along track viewing angle in degrees

    //# Size of detectors
    const fp VNIR_detector_size=13e-6; //# in meter
    const fp SWIR_detector_size=25e-6; //# in meter

    // Improvising a bit, to allow working in 3 modes:
    int i_band = iChannel;

    //# Pupil stop option
    fp pupil_stop;
    if (i_band == 4) { pupil_stop=2e-9; }
    if (i_band != 4) { pupil_stop=5e-10; }

    //# Derive the number of virtual detectors
    fp AL_nb_detectors, AC_nb_detectors;
    if (i_band <= 3) {
        AL_nb_detectors = 2.*tan(alpha_AL_max*M_PI/180.)*focal/VNIR_detector_size;
        AC_nb_detectors = IMGWIDTH;
    }

    if (i_band == 4) {
        AL_nb_detectors = 2.*tan(alpha_AL_max*M_PI/180.)*focal/SWIR_detector_size;
        AC_nb_detectors = 2.*tan(alpha_AC_max*M_PI/180.)*focal/SWIR_detector_size;
    }

    //# ;;;;;;;;;;;;;;;;;;;;;;;;
    //#  Define input images
    //# ;;;;;;;;;;;;;;;;;;;;;;;
    int image_dim_y = round(AL_nb_detectors);
    int image_dim_x = round(AC_nb_detectors);

    if (image_dim_x != IMGWIDTH || image_dim_y != IMGHEIGHT)
        debug_printf(
            LVL_PANIC,
            "Contract violation!\n"
            "Input image size %dx%d instead of %dx%d - aborting...\n",
            image_dim_x, image_dim_y, IMGWIDTH, IMGHEIGHT);

    m2d input_TOA_radiance(image_dim_y, image_dim_x);

    int i,j;

    m2d input_flux_cal(image_dim_y, image_dim_x);
    if (sInputFilename != "") {
        FILE *fpIn = fopen(sInputFilename.c_str(), "r");
        for(i=0; i<image_dim_y; i++) {
            for(j=0; j<image_dim_x; j++) {
                fp data;
#ifdef USE_DOUBLEPRECISION
                int res = fscanf(fpIn, "%lf", &data);
#else
                int res = fscanf(fpIn, "%f", &data);
#endif
                if (res != 1) {
                    debug_printf(
                        LVL_PANIC,
                        "Failed to read %s ...\n",
                        sInputFilename.c_str());
                } else
                    input_flux_cal(i,j) = data;
            }
        }
    } else {
        //# Define a bright background at radiometric level 100
        for(i=0; i<image_dim_y; i++)
            for(j=0; j<image_dim_x; j++)
                input_flux_cal(i,j) = input_TOA_radiance(i,j) + 100.;
    }

    m2d output_flux_cal(image_dim_y, image_dim_x);

    if (bBenchmark) {
        long total=0, totalSq=0, n=0;
        long minTime = 1000000;
        for(int bench=0; bench<50; bench++) {
            debug_printf(LVL_INFO, "Processing image...\n");
            long st = getTimeInMS();
            forward_model_single_psf_dual_resolution(
                output_flux_cal, // C-style: destination first param
                input_flux_cal, i_band,surf_rough_M1,surf_rough_M2, surf_rough_M3, dust_ppm, pupil_stop, swir_ghost,
                bench==49);
            if (bench>1) { // Skip the PSF calculation
                long en = getTimeInMS();
                n++;
                total += en-st;
                totalSq += (en-st)*(en-st);
                minTime = min(minTime, en-st);
            }
        }
        double variance = (double(totalSq) - total*total/double(n))/(n-1);
        printf("Execution time per image : %ld ms\n", minTime);
        printf("Average and std deviation: %5.2f +/- %2.1f ms",
               double(total)/n,
               (100.0*sqrt(variance)*n/total));
    } else {
        debug_printf(LVL_INFO, "Processing image...\n");
        forward_model_single_psf_dual_resolution(
            output_flux_cal, // C-style: destination first param
            input_flux_cal, i_band,surf_rough_M1,surf_rough_M2, surf_rough_M3, dust_ppm, pupil_stop, swir_ghost,
            true);
    }

    return 0;
}
