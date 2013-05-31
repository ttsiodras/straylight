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

#ifndef __CONFIG_H__
#define __CONFIG_H__

#include <vector>
#include <string.h>
#include <string>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <stdio.h>

#include "config.h"

// These are the sizes computed from the IDL prototype
// They must not change across subsequent
// forward_model_single_psf_dual_resolution() calls!
#define IMGHEIGHT   813
// VITO may ask us to change to 5200 pixels per scanline
#define IMGWIDTH    5271
#define KERNEL_SIZE 11

// What precision will we use -
// single or double?
#ifdef USE_DOUBLEPRECISION
typedef double fp;
typedef fp FFTWRealType;
#define FFTW_PREFIX(x) fftw_ ## x
#else
typedef float fp;
typedef fp FFTWRealType;
#define FFTW_PREFIX(x) fftwf_ ## x
#endif

// Templates for 1D and 2D matrices - initially done via STL::vector,
// but these are tightly packed, and have better behavior - cache-wise.
class m2d {
public:
    int _width, _height;
    fp *_pData;
    m2d(int y, int x): _width(x), _height(y), _pData(new fp[y*x]) {}
    ~m2d() { delete [] _pData; _pData=NULL; }
    fp& operator()(int y, int x) { return _pData[y*_width + x]; }
    fp* getLine(int y) { return &_pData[y*_width]; }
    void reset() { memset(_pData, 0, sizeof(fp)*_width*_height); }
private:
    m2d(const m2d&);
    m2d& operator=(const m2d&);
};
class m1d {
public:
    int _width;
    fp *_pData;
    m1d(int x): _width(x), _pData(new fp[x]()) {}
    ~m1d() { delete [] _pData; _pData=NULL; }
    fp& operator[](int x) { return _pData[x]; }
    void reset() { memset(_pData, 0, sizeof(fp)*_width); }
    operator fp*() { return _pData; }
private:
    m1d(const m1d&);
    m1d& operator=(const m1d&);
};

#endif
