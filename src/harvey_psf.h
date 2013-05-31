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

#ifndef __harvey_psf_h__
#define __harvey_psf_h__

#include "configStraylight.h"

void harvey_psf(
    m2d& psf,
    m2d& radius,
    int channel,
    fp surface_roughness_mirror1,
    fp surface_roughness_mirror2,
    fp surface_roughness_mirror3,
    bool INCIDENCE_ANGLE_CENTER_FOV = false);

#endif
