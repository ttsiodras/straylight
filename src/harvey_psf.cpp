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

#include "harvey_psf.h"
#include "harvey_brdf.h"
#include "tis_surface_scattering_harvey.h"

//+
// NAME:
// HARVEY_PSF
//
// PURPOSE:
// Return the PSF induced by the surface roughness of the 3 mirrors in an array of user defined dimensions 
// following a variant of the Harvey model for diamond turned aluminium samples
// Surface roughness and wavelength are also input to the PSF model

// CATEGORY:
// Optics
//
// CALLING SEQUENCE:
// psf=HARVEY_PSF(radius, channel, surface_roughness_mirror1, surface_roughness_mirror2 ,surface_roughness_mirror3)
//
// INPUTS:
// - radius: an array containing the radius to the Gaussian image in the focal plane in mm
// - channel: [1, 4]
// - surface_roughness_M1 to M3: surface roughness in meter of respectively M1 to M3
//
// OPTIONAL INPUT PARAMETERS:
// - KEYWORD: INCIDENCE_ANGLE_CENTER_FOV
//            If set, this keyword set the incidence angles used for the PSF computation on each mirror to the values for the 
//            chief ray corresponding to a source in the center of the FOV
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
// Model mentioned and described in Taccola in a TN on Proba-V straylight. Largely inspired from Taracola code. 
// PSF model originally developed by Harvey  
// !!! give ref
//

// MODIFICATION HISTORY:
// Written mbouvet@esa.int , 2011 - JULY 25

using namespace std;

void harvey_psf(
    m2d& psf,
    m2d& radius,
    int channel,
    fp surface_roughness_mirror1, 
    fp surface_roughness_mirror2,
    fp surface_roughness_mirror3,
    bool INCIDENCE_ANGLE_CENTER_FOV)
{
    int i,j;

//    INCIDENCE_ANGLE_CENTER_FOV=incidence_angle_center_FOV

// Altitude
//h=820.0

//focal lenght
//F=110.0

//numerical aperture (F#7)
    const fp Fnum = 7.0;
    const fp na=1.0/(2.0*Fnum);

    const fp wl[] = {0.45*1e-6, 0.65*1e-6, 0.825*1e-6, 1.6*1e-6};

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
    const fp b_6nm_blue=82.0;
    const fp s_6nm_blue=-2.4;
    const fp l_6nm_blue=0.005;
    const fp m_6nm_blue=0.8;
    const fp n_6nm_blue=0.8;

// Red band
    const fp b_6nm_red=82.0;
    const fp s_6nm_red=-2.0;
    const fp l_6nm_red=0.002;
    const fp m_6nm_red=0.0;
    const fp n_6nm_red=0.0;

// NIR band
    const fp b_6nm_nir=30.0;
    const fp s_6nm_nir=-2.4;
    const fp l_6nm_nir=0.0045;
    const fp m_6nm_nir=0.0;
    const fp n_6nm_nir=0.0;

// SWIR band
    const fp b_6nm_swir=5.0;
    const fp s_6nm_swir=-2.5;
    const fp l_6nm_swir=0.006;
    const fp m_6nm_swir=0.0;
    const fp n_6nm_swir=0.0;

// Put all previous values in a array for all bands
    const fp b_6nm[]={b_6nm_blue,b_6nm_red,b_6nm_nir,b_6nm_swir};
    const fp s_6nm[]={s_6nm_blue,s_6nm_red,s_6nm_nir,s_6nm_swir};
    const fp l_6nm[]={l_6nm_blue,l_6nm_red,l_6nm_nir,l_6nm_swir};
    const fp m_6nm[]={m_6nm_blue,m_6nm_red,m_6nm_nir,m_6nm_swir};
    const fp n_6nm[]={n_6nm_blue,n_6nm_red,n_6nm_nir,n_6nm_swir};

// 
// Total integrated scattering for 6 nm surface roughness mirrors assuming a 0 degree incidence angle
// 
    fp TIS_6nm[4];
    TIS_6nm[0] = tis_surface_scattering_harvey(1,0., 6e-9);
    TIS_6nm[1] = tis_surface_scattering_harvey(2,0., 6e-9);
    TIS_6nm[2] = tis_surface_scattering_harvey(3,0., 6e-9);
    TIS_6nm[3] = tis_surface_scattering_harvey(4,0., 6e-9);

// Compute the TIS for the input surface roughness

    fp surface_roughness[3];
    surface_roughness[0] = (surface_roughness_mirror1);
    surface_roughness[1] = (surface_roughness_mirror2);
    surface_roughness[2] = (surface_roughness_mirror3);

// Compute the b parameter corresponding to the surface roughness
    fp b_surface_roughness[3];
    for(int i=0; i<3; i++) {
	fp tmpTIS_surface_roughness = pow(4. * M_PI * surface_roughness[i]/wl[channel-1], 2);
	b_surface_roughness[i] = b_6nm[channel-1]*tmpTIS_surface_roughness/TIS_6nm[channel-1];
    }

//;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
// Compute the direct part of the PSF
//;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

// When the keyword INCIDENCE_ANGLE_CENTER_FOV is set, we use the center of the FOV and the correspondint incidence 
// angles to compute the TIS
    fp i_angles_center_FOV[] = {17.92, 23.03, 11.57}; // in degrees
    fp theta0[3];

    if (INCIDENCE_ANGLE_CENTER_FOV) {
       	theta0[0] = i_angles_center_FOV[0]*M_PI/180.;
       	theta0[1] = i_angles_center_FOV[1]*M_PI/180.;
       	theta0[2] = i_angles_center_FOV[2]*M_PI/180.;
    } else {
	theta0[0]=0.;
	theta0[1]=0.;
	theta0[2]=0.;
    }

// Compute the TIS
    fp TIS_surface_roughness[3];
    for(int i=0; i<3; i++)
	TIS_surface_roughness[i] = tis_surface_scattering_harvey(channel, theta0[i], surface_roughness[i]);

//;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
// Compute the diffuse part of the PSF
//;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
// Following Peterson et al. 
//;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

// Entrance irradiance - This values has no impact on the PSF
    const fp E_ent=1.; // in W.m-2

// // Computation of the Harvey brdf
//     m2d brdf_M1;
//     m2d radiusTimesNaDivMaPlusTheta0;
//     for(unsigned i=0; i<radius.size(); i++)
// 	for(unsigned j=0; j<radius[0].size(); j++)
// 	    radiusTimesNaDivMaPlusTheta0(i,j) = radius(i,j)*na/mirror_aperture[0]+theta0[0];
//     harvey_brdf( brdf_M1, radiusTimesNaDivMaPlusTheta0, theta0[0], b_surface_roughness[0],
// 	s_6nm[channel-1], l_6nm[channel-1], m_6nm[channel-1], n_6nm[channel-1]);
// 
// // Irradiance in focal plane
//     m2d psf_M1(brdf_M1.size());
//     for(unsigned i=0; i<brdf_M1.size(); i++) {
// 	psf_M1[i].resize(brdf_M1[0].size());
// 	for(unsigned j=0; j<brdf_M1[0].size(); j++) {
// 	    fp irr_distrib_focal_M1 = 
// 		E_ent*M_PI*pow(mirror_aperture[0],2)*brdf_M1[i][j]*pow(na,2)*pow(1./mirror_aperture[0],2);
// // Computation of the radiant power in the focal plane at each pixel
// 	    fp power_focal_M1 = irr_distrib_focal_M1*detector_size*detector_size;
// // Computation of the normalised power distribution in focal plane 
// 	    fp norm_power_distrib_focal_M1 = power_focal_M1/(E_ent*M_PI*pow(mirror_aperture[0],2));
// 	    psf_M1[i][j] = norm_power_distrib_focal_M1;
// 	}
//     }

// Computation of the Harvey brdf
    psf.reset();
    m2d radiusTimesNaDivMaPlusTheta0(radius._height, radius._width);
    for(int phase=0; phase<3; phase++) {
	for(i=0; i<radius._height; i++) {
	    for(j=0; j<radius._width; j++)
		radiusTimesNaDivMaPlusTheta0(i,j) = radius(i,j)*na/mirror_aperture[phase]+theta0[phase];
	}
	m2d brdf_M(radius._height, radius._width);
	harvey_brdf( brdf_M, radiusTimesNaDivMaPlusTheta0, theta0[phase], b_surface_roughness[phase],
	    s_6nm[channel-1], l_6nm[channel-1], m_6nm[channel-1], n_6nm[channel-1]);

// Irradiance in focal plane
	fp common = E_ent*M_PI*pow(mirror_aperture[0],2);
	for(i=0; i<brdf_M._height; i++) {
	    for(j=0; j<brdf_M._width; j++) {
		fp irr_distrib_focal_M = common*brdf_M(i,j)*pow(na,2)*pow(1./mirror_aperture[phase],2);
// Computation of the radiant power in the focal plane at each pixel
		fp power_focal_M = irr_distrib_focal_M*detector_size*detector_size;
// Computation of the normalised power distribution in focal plane 
		fp norm_power_distrib_focal_M = power_focal_M/common;
// Add up all mirror contibutions due to surface roughness
		psf(i,j) += norm_power_distrib_focal_M;
	    }
	}
    }

// Add up the direct part of the PSF
    fp dummy_var = 1.e20;
    int imin=0, jmin=0;
    for(i=0; i<radius._height; i++) {
	for(j=0; j<radius._width; j++) {
	    if (radius(i,j) < dummy_var) {
		imin = i;
		jmin = j;
		dummy_var = radius(i,j);
	    }
	}
    }

    if (dummy_var == 0.)
	psf(imin,jmin) = (1.-TIS_surface_roughness[0])*(1.-TIS_surface_roughness[1])*(1.-TIS_surface_roughness[2]);
}
