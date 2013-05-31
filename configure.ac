AC_INIT([strayLight], [1.1c], [ttsiodras@gmail.com])

AC_CONFIG_HEADERS([src/config.h])
AC_CONFIG_MACRO_DIR([ac-macros])
AC_CONFIG_AUX_DIR([aux-dir])

########################
# Configure arguments
########################
AC_ARG_ENABLE(debug,
    [AS_HELP_STRING([--enable-debug],[Compile in debug mode (no CUDA, no OpenMP, no optimizations).])],
    [enable_debug="$enableval"],
    [enable_debug="no"])

AC_ARG_ENABLE(eigen,
    [AS_HELP_STRING([--enable-eigen],[use the Eigen SSE library for convolutions (automatic, use --disable-openmp to disable it])],
    [enable_eigen="$enableval"],
    [enable_eigen="yes"])

AC_ARG_ENABLE(double,
    [AS_HELP_STRING([--enable-double],[use double precision for calculations])],
    [enable_double="$enableval"],
    [enable_double="no"])

AC_ARG_ENABLE(openmp,
    [AS_HELP_STRING([--enable-openmp],[use OpenMP Multithreading if available (automatic, use --disable-openmp to disable it)])],
    [enable_openmp="$enableval"],
    [enable_openmp="yes"])

AC_ARG_ENABLE(cuda,
    [AS_HELP_STRING([--enable-cuda],[use CUDA if available (automatic, use --disable-cuda to disable it)])],
    [enable_cuda="$enableval"],
    [enable_cuda="yes"])

AC_ARG_ENABLE(cache,
    [AS_HELP_STRING([--enable-cache],[cache and re-use calculated PSF data across calls (automatic, use --disable-cache to disable it)])],
    [enable_cache="$enableval"],
    [enable_cache="yes"])

AC_ARG_WITH([cudaSDK],
    [  --with-cudaSDK=PATH        prefix where CUDA SDK is installed])

########################
# Detect the canonical host and target build environment
########################
AC_CANONICAL_HOST
AC_CANONICAL_TARGET

AC_LANG(C++)

AM_INIT_AUTOMAKE([-Wall -Werror])
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])

AM_PROG_CC_C_O

if test x"${CXXFLAGS}" = x ; then
    CXXFLAGS="-O3 -g -Wall"
fi

AC_PROG_CXX

AC_CHECK_FUNCS([atexit])
AC_CHECK_FUNCS([clock_gettime])
AC_CHECK_FUNCS([memset])
AC_CHECK_FUNCS([pow])
AC_CHECK_FUNCS([select])
AC_CHECK_FUNCS([sqrt])
AC_CHECK_HEADERS([sys/time.h])
AC_CHECK_HEADER_STDBOOL
AC_CHECK_TYPES([ptrdiff_t])
AC_C_INLINE
AC_C_RESTRICT
AC_FUNC_REALLOC
AC_HEADER_MAJOR
AC_TYPE_INT32_T
AC_TYPE_SIZE_T

AC_CHECK_HEADERS([stdlib.h string.h])
AC_FUNC_MALLOC
AC_FUNC_VPRINTF
AC_HEADER_STDC

if test x"${enable_cache}" = xyes ; then
    AC_DEFINE([USE_PSFCACHE], 1, [Define this to reuse PSF data across calls.])
fi

#######################################################
# Check for double precision and use appropriate FFTW3
#######################################################
PKG_CHECK_MODULES([FFTW], [fftw3],,AC_MSG_ERROR("fftw3 is required - e.g. apt-get install libfftw3-dev"))
if test x"${enable_double}" = xyes ; then
    AC_DEFINE([USE_DOUBLEPRECISION], 1, [Define this to use double precision.])
else
    FFTW_LIBS=$(echo "${FFTW_LIBS}" | sed 's,fftw3,fftw3f,g')
fi
AC_SUBST(FFTW_LIBS)

########################
# Check for OpenMP
########################
OPENMP_CXXFLAGS=""
OPENMP_LIBS=""
HAVE_OPENMP=no
if test x"${enable_openmp}" = xyes ; then
    AX_OPENMP([HAVE_OPENMP=yes])
    if test x"${HAVE_OPENMP}" != xyes ; then
        AC_MSG_RESULT($GOODGCC)
        AC_MSG_NOTICE([*************************************************])
        AC_MSG_NOTICE([**   Your compiler is missing OpenMP support   **])
        AC_MSG_NOTICE([**     StrayLight will run MUCH slower...      **])
        AC_MSG_NOTICE([*************************************************])
        HAVE_OPENMP=no
    else
        AC_MSG_CHECKING(for GCC maturity in OpenMP support)
        AC_RUN_IFELSE([AC_LANG_SOURCE([#include <stdlib.h>
int main(void) {
#if defined(__GNUC__)
#if __GNUC__ < 4 || ((__GNUC__ == 4) && (__GNUC_MINOR__ < 3))
    return -1;
#endif
#endif
    return 0;
}
        ])], [GOODGCC=yes], [GOODGCC=no])
        if test x"${GOODGCC}" = xyes ; then
            AC_MSG_RESULT($GOODGCC)
            if test x"${enable_debug}" != xyes ; then
                AC_DEFINE([USE_OPENMP], 1, [Define this to use OpenMP.])
                OPENMP_LIBS=${OPENMP_CXXFLAGS}
                CPPFLAGS="$CPPFLAGS $OPENMP_CXXFLAGS"
            else
                AC_MSG_NOTICE([**********************************************************])
                AC_MSG_NOTICE([*       Debug mode - disabling OpenMP and SSE            *])
                AC_MSG_NOTICE([**********************************************************])
            fi
        else
            AC_MSG_RESULT($GOODGCC)
            AC_MSG_NOTICE([**********************************************************])
            AC_MSG_NOTICE([** You have an old version of GCC, that has OpenMP bugs **])
            AC_MSG_NOTICE([**        OpenMP disabled to avoid crashes...           **])
            AC_MSG_NOTICE([**         StrayLight will run MUCH slower...           **])
            AC_MSG_NOTICE([**********************************************************])
            HAVE_OPENMP=no
        fi
    fi
else
    AC_MSG_NOTICE([*********************************************])
    AC_MSG_NOTICE([*      Instructed to disable OpenMP.        *])
    AC_MSG_NOTICE([*    StrayLight will run MUCH slower...     *])
    AC_MSG_NOTICE([*********************************************])
fi
AC_SUBST(OPENMP_LIBS)

CUDA_CFLAGS=""
CUDA_LIBS=""
NVCC=""
NVCCFLAGS=""
HAVE_CUDA=no
##########################################
# Check for CUDA (if not disabled by user)
##########################################
if test x"${enable_cuda}" = xyes ; then

    if test "x$with_cudaSDK" != x; then
        CUDA_CFLAGS="-I$with_cudaSDK/include"
        CUDA_LIBS="-L$with_cudaSDK/lib -lcudart"
        NVCC="$with_cudaSDK/bin/nvcc"
    else
        # Assume it is installed in global directories (e.g. via Debian's CUDA packages)
        CUDA_CFLAGS=""
        CUDA_LIBS="-lcudart"
        NVCC="nvcc"
    fi
    NVCCFLAGS="-I . -O3 -use_fast_math --compiler-options -fno-inline -arch sm_21 "
    
    OLDCFLAGS="$CFLAGS"
    OLDCPPFLAGS="$CPPFLAGS"
    CFLAGS="$CUDA_CFLAGS"
    CPPFLAGS="$CUDA_CFLAGS"
    #AC_CHECK_HEADER([cuda.h],,[AC_MSG_ERROR([Sorry, CUDA SDK appears to be missing])])
    AC_MSG_CHECKING(for CUDA)
    AC_CHECK_HEADER([cuda.h],[HAVE_CUDA=yes],[HAVE_CUDA=no])
    CFLAGS="$OLDCFLAGS"
    CPPFLAGS="$OLDCPPFLAGS"
    if test x"${HAVE_CUDA}" = xyes ; then
        AC_DEFINE([USE_CUDA_GPU], 1, [Define this to use CUDA GPU.])
    else
        AC_MSG_WARN([[*****************************************************]])
        AC_MSG_WARN([[* No CUDA detected - StrayLight can run much faster *]])
        AC_MSG_WARN([[*      Perhaps you need to use --with-cudaSDK ?     *]])
        AC_MSG_WARN([[*****************************************************]])
        CUDA_CFLAGS=""
        CUDA_LIBS=""
    fi
else
    AC_MSG_NOTICE([*********************************************])
    AC_MSG_NOTICE([*      Instructed to disable CUDA.          *])
    AC_MSG_NOTICE([*********************************************])
fi
AM_CONDITIONAL([CUDA_GPU], [test x$HAVE_CUDA = xyes])
AC_SUBST(CUDA_CFLAGS)
AC_SUBST(CUDA_LIBS)
AC_SUBST(NVCC)
AC_SUBST(NVCCFLAGS)

CXXFLAGS+=" ${CUDA_CFLAGS}"
CXXFLAGS+=" ${FFTW_CFLAGS}"

#########################################################################
# Use Eigen, if no CUDA selected (or available) and Eigen is not disabled
#########################################################################
if test x"${HAVE_CUDA}" = xyes ; then
    # Never use Eigen if CUDA is in use
    AC_MSG_NOTICE([Using CUDA for convolution, skipping Eigen check])
else
    if test x"${enable_eigen}" = xyes ; then
        if test x"${enable_double}" = xyes ; then
            AC_MSG_WARN([[*****************************************************]])
            AC_MSG_WARN([[*   Eigen cannot be used with double precision      *]])
            AC_MSG_WARN([[*****************************************************]])
        else
            AC_DEFINE([USE_EIGEN], 1, [Define this to use the Eigen SSE library for convolutions.])
            AC_MSG_NOTICE([*********************************************************])
            AC_MSG_NOTICE([* No CUDA available, will compute convolution via Eigen *])
            AC_MSG_NOTICE([*********************************************************])
        fi
    else
        AC_MSG_NOTICE([*********************************************])
        AC_MSG_NOTICE([*      Instructed to disable Eigen.         *])
        AC_MSG_NOTICE([*********************************************])
    fi
fi

########################
# Best optimization flags for our compiler
########################
if test x"${enable_debug}" = xyes ; then
        AC_MSG_NOTICE([Disabling optimizations])
        CXXFLAGS=`echo "$CXXFLAGS" | sed 's,-O3,,g'`
else
        AX_CXXFLAGS_GCC_OPTION(-Wall)
        AX_CXXFLAGS_GCC_OPTION(-Wextra)
        AX_CXXFLAGS_GCC_OPTION(-pedantic)
        AX_CXXFLAGS_GCC_OPTION(-fomit-frame-pointer)
        #AX_CXXFLAGS_GCC_OPTION(-ffast-math)
        #AX_CXXFLAGS_GCC_OPTION(-funsafe-math-optimizations)
        AX_CXXFLAGS_GCC_OPTION(-mtune=native)

# GCC support for -flto is still buggy, internal compiler error triggered...
#        AX_CXXFLAGS_GCC_OPTION(-flto)
#        AX_CHECK_LINKER_FLAGS(-fwhole-program,[WHOLEPROGRAM="-fwhole-program"],[])
#        AC_SUBST(WHOLEPROGRAM)

########################
# SIMD detection
########################

        AC_MSG_CHECKING(for SSE)
        AC_RUN_IFELSE([AC_LANG_SOURCE([#include <stdlib.h>
#include <signal.h>
void catchme(int) { exit(1); }
int main(void) {
signal(SIGILL, catchme);
__asm__ volatile ("xorps %%xmm0, %%xmm0":::"memory"); return 0;
}
        ])], [HAVESSE=yes], [HAVESSE=no])
        AC_MSG_RESULT($HAVESSE)
        if test x"${HAVESSE}" = xyes ; then
            AC_DEFINE([SIMD_SSE], 1, [Define this to use SSE intrinsics.])
        fi

        AC_MSG_CHECKING(for SSE2)
        AC_RUN_IFELSE([AC_LANG_SOURCE([#include <stdlib.h>
#include <signal.h>
void catchme(int) { exit(1); }
int main(void) {
signal(SIGILL, catchme);
__asm__ volatile ("xorpd %%xmm0, %%xmm0":::"memory"); return 0;
}
        ])], [HAVESSE2=yes], [HAVESSE2=no])
        AC_MSG_RESULT($HAVESSE2)

        AC_MSG_CHECKING(for SSSE3)
        AC_RUN_IFELSE([AC_LANG_SOURCE([#include <stdlib.h>
#include <signal.h>
void catchme(int) { exit(1); }
int main(void) {
signal(SIGILL, catchme);
__asm__ volatile ("pabsd %%xmm0, %%xmm0":::"memory"); return 0;
}
        ])], [HAVESSSE3=yes], [HAVESSSE3=no])
        AC_MSG_RESULT($HAVESSSE3)

        if test x"${HAVESSE}" = xyes ; then
            AX_CXXFLAGS_GCC_OPTION(-msse)
            # Dont use mrecip - we need accuracy
            #AX_CXXFLAGS_GCC_OPTION(-mrecip)
            AX_CXXFLAGS_GCC_OPTION(-mfpmath=sse)
        fi
        if test x"${HAVESSE2}" = xyes ; then
            AX_CXXFLAGS_GCC_OPTION(-msse2)
        fi
        if test x"${HAVESSSE3}" = xyes ; then
            AX_CXXFLAGS_GCC_OPTION(-mssse3)
        fi
        CXXFLAGS="$CXXFLAGS -DNDEBUG"
fi

#
# Leaving detectors for OpenGL and SDL here, for potential future use.
#
########################
# OPENGL
########################
#
## Figure out which OpenGL and GLU libraries to use
#case "$target" in
#    *-*-cygwin* | *-*-mingw32*)
#        SYS_GL_LIBS="-lopengl32 -lglu32 -lglut"
#        ;;
#    *-*-beos*)
#        SYS_GL_LIBS="-lGL -lGLU -lGLEW -lglut"
#        ;;
#    *-*-aix*)
#        if test x$ac_cv_prog_gcc = xyes; then
#            CFLAGS="-mthreads"
#        fi
#        ;;
#    *-*-darwin*)
#       CPPFLAGS="$CPPFLAGS -I/usr/X11R6/include"
#       SYS_GL_LIBS="-L/usr/X11R6/lib -lGL -lGLU -lGLEW -lglut"
#        ;;
#    *)
#        AC_PATH_X
#        AC_PATH_XTRA
#        if test x$have_x = xyes; then
#            CPPFLAGS="$CPPFLAGS $X_CFLAGS"
#            SYS_GL_LIBS="$X_LIBS -lGL -lGLU -lGLEW -lglut"
#        else
#            AC_MSG_ERROR([*** No X detected!])
#        fi
#        ;;
#esac
#
## Check for OpenGL
#AC_MSG_CHECKING(for OpenGL GLUT and GLEW support)
#HAVE_OPENGL=no
#AC_TRY_COMPILE([
# #include <GL/glew.h>
# #include <GL/gl.h>
# #include <GL/glut.h>
#],[
#],[
#HAVE_OPENGL=yes
#])
#AC_MSG_RESULT($HAVE_OPENGL)
#if test x$HAVE_OPENGL != xyes; then
#    AC_MSG_ERROR([*** OpenGL GLEW and GLUT must be installed!])
#fi
#AC_SUBST(SYS_GL_LIBS)

########################
# SDL
########################
#
# Check for SDL (minimum: 1.2.0)
#SDL_VERSION=1.2.0
#AM_PATH_SDL($SDL_VERSION, :,
#            AC_MSG_ERROR([*** SDL version $SDL_VERSION or later was not found!]))
#AC_SUBST(SDL_CFLAGS)
#AC_SUBST(SDL_LIBS)

####################################
# Emit all the generated Makefiles
####################################
AC_CONFIG_FILES([Makefile src/Makefile]) 
AC_OUTPUT
