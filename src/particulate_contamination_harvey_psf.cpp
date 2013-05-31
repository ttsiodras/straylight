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

#include <math.h>

#include "particulate_contamination_harvey_psf.h"

//+
// NAME:
// MODIFIED_HARVEY_BRDF
//
// PURPOSE:
// Return the BRDF following a variant of the Harvey model 
//
// CATEGORY:
// Optics
//
// CALLING SEQUENCE:
// 
//
// INPUTS:
// theta: scattering angle in radian with respect to the normal of the mirror
// theta0: incidence angle in radian of main ray wrt to the normal to the surface of the mirror
// b: model parameter
// s: model parameter
// l: model parameter
//
// 
//
// OPTIONAL INPUT PARAMETERS:
// 
// OUTPUTS:
// 
// COMMON BLOCKS:
// None.
//
// SIDE EFFECTS:
// None.
//
// RESTRICTIONS:
// None.
//
// PROCEDURE:
// Originally developed by Harvey. 
//
// MODIFICATION HISTORY:
// Written mbouvet@esa.int , 2011 - AUGUST 15

fp modified_harvey_brdf(fp theta, fp theta0, fp b, fp s, fp l)
{
    return b*pow((1.+pow((sin(theta)-sin(theta0))/l,2)),s/2.);
}

//+
// NAME:
// PARTICULATE_CONTAMINATION_HARVEY_PSF
//
// PURPOSE:
// Return the PSF induced by the particulate contamination of the 3 mirrors in an array of user defined dimensions
// following a variant of the Harvey model. The model paramters were obtained from fitting Mie calculation with a
// 1000 ppm contamination
//
// CATEGORY:
// Optics
//
// CALLING SEQUENCE:
// psf=particulate_CONTAMINATION_HARVEY_PSF(radius, channel)
//
// INPUTS:
// - radius: an array containing the radius to the Gaussian image in the focal plane in mm
// - channel: [1, 4]
//
// OPTIONAL INPUT PARAMETERS:
//
// OUTPUTS:
// - an array of same size than radius containing the PSF with both the direct component (1-TIS_M1-TIS_M2-TIS_M3) and the
//   diffuse component
//
// COMMON BLOCKS:
// None.
//
// SIDE EFFECTS:
// None.
//
// RESTRICTIONS:
// None.
//
// PROCEDURE:
// Model originally developed by Harvey
//
//

// MODIFICATION HISTORY:
// Written mbouvet@esa.int , 2011 - JULY 25

void particulate_contamination_harvey_psf(m2d& psf_mirror_dust, m2d& radius, int channel)
{
    // Altitude
    //h=820.0

    //focal lenght
    //F=110.0

    //numerical aperture (F#7)
    const fp Fnum=7.0;
    const fp na=1.0/(2.0*Fnum);

    //const fp wl[] = {0.45*1e-6, 0.65*1e-6, 0.825*1e-6, 1.6*1e-6};

    //optical trasmission
    //const fp optics_transmission=1.0;

    // Size of detectors
    const fp VNIR_detector_size=13e-6; // in meter
    const fp SWIR_detector_size=25e-6; // in meter
    fp detector_size = 0.;
    if (channel <= 3)
	detector_size = VNIR_detector_size;
    if (channel == 4)
       	detector_size = SWIR_detector_size;

    // !!!!
    //size of the instrument aperture at each mirror in meter: from Taracola code => I don't understand where these values come from
    // !!!!
    const fp mirror_aperture[] = {9.5*6.0/Fnum*1e-3, 5.05*6.0/Fnum*1e-3, 6.85*6.0/Fnum*1e-3};

    // Define the parameters for the BRDF of the mirror for each spectral band for a surface roughness of 6 nm
    // These are values given to Taracola by CSL

    // Blue band
    const fp b_particulate_blue_1=8.13;
    const fp s_particulate_blue_1=-2.17;
    const fp l_particulate_blue_1=0.0026;
    const fp b_particulate_blue_2=0.00244;
    const fp s_particulate_blue_2=-0.881;
    const fp l_particulate_blue_2=0.0431513;

    // Red band
    const fp b_particulate_red_1=3.93;
    const fp s_particulate_red_1=-2.35;
    const fp l_particulate_red_1=0.00411;
    const fp b_particulate_red_2=0.00429;
    const fp s_particulate_red_2=-1.11;
    const fp l_particulate_red_2=0.0431513;


    // NIR band
    const fp b_particulate_nir_1=2.63;
    const fp s_particulate_nir_1=-2.23;
    const fp l_particulate_nir_1=0.00472;
    const fp b_particulate_nir_2=0.00441;
    const fp s_particulate_nir_2=-1.06467;
    const fp l_particulate_nir_2=0.0431513;


    // SWIR band
    const fp b_particulate_swir_1=0.557;
    const fp s_particulate_swir_1=-1.36716;
    const fp l_particulate_swir_1=0.0025166;
    const fp b_particulate_swir_2=0.37724;
    const fp s_particulate_swir_2=-2.40281;
    const fp l_particulate_swir_2=0.0116923;


    // Put all previous values in a array for all bands
    const fp b_particulate_1[] = {b_particulate_blue_1,b_particulate_red_1,b_particulate_nir_1,b_particulate_swir_1};
    const fp s_particulate_1[] = {s_particulate_blue_1,s_particulate_red_1,s_particulate_nir_1,s_particulate_swir_1};
    const fp l_particulate_1[] = {l_particulate_blue_1,l_particulate_red_1,l_particulate_nir_1,l_particulate_swir_1};
    const fp b_particulate_2[] = {b_particulate_blue_2,b_particulate_red_2,b_particulate_nir_2,b_particulate_swir_2};
    const fp s_particulate_2[] = {s_particulate_blue_2,s_particulate_red_2,s_particulate_nir_2,s_particulate_swir_2};
    const fp l_particulate_2[] = {l_particulate_blue_2,l_particulate_red_2,l_particulate_nir_2,l_particulate_swir_2};

    //; !!! This defines the incidence angle of the chief ray on the mirror - This should actually be variable in the FOV !!!
    //; The value of the TIS should also depend on this angle
    const fp theta0=0.;

    //;;;;;;;;
    // Compute the TIS
    //;;;;;;;;;

    fp TIS_dust = 0.;
    for(int i=0; i<1000; i++) {
	fp theta_rd = float(i)/1000.*M_PI/2.+1e-6;
	fp brdf_particulate = 
	    modified_harvey_brdf(
		theta_rd, theta0, 
		b_particulate_1[channel-1],
		s_particulate_1[channel-1],
		l_particulate_1[channel-1]) 
	    +
	    modified_harvey_brdf(
		theta_rd, theta0,  
		b_particulate_2[channel-1],
		s_particulate_2[channel-1],
		l_particulate_2[channel-1]);
	TIS_dust += 2.*M_PI*brdf_particulate*cos(theta_rd)*sin(theta_rd)*1./1000.*M_PI/2.;
    }

    // Compute the diffuse part of the PSF
    const fp E_ent=1.; // in W.m-2

    psf_mirror_dust.reset();
    for(int phase=0; phase<3; phase++) {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
	for(int i=0; i<radius._height; i++) {
	    for(int j=0; j<radius._width; j++) {
		// !!!!!!!!!!!!!!!!!
		// Following Peterson et al. we have ...
		// !!!!!!!!!!!!!
		// Computation of the Harvey brdf
		fp radiusTimesNaDivMa = radius(i,j)*na/mirror_aperture[phase];
		fp brdf_M = 
		    modified_harvey_brdf(
			radiusTimesNaDivMa,  
			theta0,
			b_particulate_1[channel-1],
			s_particulate_1[channel-1],
			l_particulate_1[channel-1])
		    +
		    modified_harvey_brdf(
			radiusTimesNaDivMa,  
			theta0,
			b_particulate_2[channel-1],
			s_particulate_2[channel-1],
			l_particulate_2[channel-1]);
		// Irradiance in focal plane
		fp irr_distrib_focal_M = E_ent*M_PI*pow(mirror_aperture[0],2)*brdf_M*pow(na,2)*pow(1./mirror_aperture[phase],2);

		// Computation of the radiant power in the focal plane at each pixel
		fp power_focal_M = irr_distrib_focal_M*pow(detector_size,2);

		// Computation of the normalised power distribution in focal plane
		fp norm_power_distrib_focal_M = power_focal_M/(E_ent*M_PI*pow(mirror_aperture[0],2));

		// Add up all mirror contibutions due to surface roughness
		psf_mirror_dust(i,j) += norm_power_distrib_focal_M;
	    }
	}
    }

    // Add up the direct part of the PSF
    fp dummy_var = 1.e20;
    int imin=0, jmin=0;
    for(int i=0; i<radius._height; i++) {
	for(int j=0; j<radius._width; j++) {
	    if (radius(i,j) < dummy_var) {
		imin = i;
		jmin = j;
		dummy_var = radius(i,j);
	    }
	}
    }

    if (dummy_var == 0.)
	psf_mirror_dust(imin,jmin) = pow(1.-TIS_dust,3);
}
