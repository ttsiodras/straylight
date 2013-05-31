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

#ifndef __forward_model_single_psf_dual_resolution__
#define __forward_model_single_psf_dual_resolution__

#include "configStraylight.h"

void forward_model_single_psf_dual_resolution(
    m2d& output_image, 
    m2d& input_image,
    int channel, 
    fp surface_roughness_M1, 
    fp surface_roughness_M2,
    fp surface_roughness_M3, 
    int ppm_dust, 
    fp pupil_stop, 
    bool swir_ghost,
    bool logStages=false);

#endif
