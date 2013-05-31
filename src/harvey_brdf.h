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

#ifndef __HARVEY_BRDF_H__
#define __HARVEY_BRDF_H__

#include "configStraylight.h"

fp harvey_brdf(fp theta, fp theta0, fp b, fp s, fp l, fp m, fp n);
void harvey_brdf(
    m2d& result,
    m2d& thetaArg, 
    fp theta0, fp b, fp s, fp l, fp m, fp n);

#endif
