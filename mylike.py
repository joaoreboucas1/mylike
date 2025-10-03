"""
.. module:: likelihoods.gaussian_mixture

:Synopsis: Gaussian mixture likelihood
:Author: Jesus Torrado

"""
# Global
from typing import Any
import numpy as np
from scipy.stats import norm, uniform, random_correlation
from scipy.special import logsumexp

# Local
from cobaya.likelihood import Likelihood
from cobaya.log import LoggedError
from cobaya.mpi import share_mpi, is_main_process
from cobaya.typing import InputDict, Union, Sequence
from cobaya.functions import inverse_cholesky

import pandas as pd
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
import camb, os, h5py
from astropy.io import fits

# SNe Functions
c = 299792.458 # [km/s]
ns = 0.96
nmod = 628

class MyLike(Likelihood):
    """
    Gaussian likelihood.
    """
    file_base_name = 'mylike'

    # yaml variables
    mean: float
    stdev: float
    
    #input_params_prefix: str
    #output_params_prefix: str

    def initialize(self):
        self.log.info("Initializing")
        super().initialize()
        self.mvecc, self.zvec_CMB, self.zvec_HEL, self.rdirs, self.cov_pan_fil = Pantheon_data()
        self._input_params_order = self.input_params
        self.gaussian = norm(loc=self.mean, scale=self.stdev)

    def get_requirements(self):
        return {
            "H0": None,
            "omegam": None,
            "sigma8": None,
            "comoving_radial_distance": {"z": self.zvec_CMB},
            "Pk_interpolator": {
                "z": [0], "k_max": 10, "nonlinear": True,
                "vars_pairs": [("delta_tot", "delta_tot")]},
            "Cl": {"tt": 0}
        }

    def initialize_with_params(self):
        """
        Initializes the gaussian distributions.
        """
        self.log.info("Initializing")
        self._input_params_order = self._input_params_order or self.input_params
        self.gaussian = norm(loc=self.mean, scale=self.stdev)

    def logp(self, **params_values):
        # OL, Om, Ob, sig8, h0, Mag, gamma, sigv = theta
        #################### theory - model (Astropy)
        # cosmo = LambdaCDM(H0=h0*100, Om0=Om, Ode0=OL, Ob0=Ob)

        sigma8 = self.provider.get_param('sigma8')
        self.log.info(f"sigma8 = {sigma8}")
        return -(sigma8 - self.mean)**2/self.stdev**2/2
        
        h = self.provider.get_param('h')
        Mag = self.provider.get_param('Mag')
        # lum_dis = cosmo.comoving_transverse_distance(zvec_CMB).value*h0*100/c*(1+zvec_HEL)
        lum_dis = self.provider.get_comoving_radial_distance(self.zvec_CMB)*h*100/c*(1+self.zvec_HEL)
        vecdif = self.mvecc - 5*np.log10(lum_dis) - (Mag - 5*np.log10(100*h/c) + 25)


        # OL, Om, Ob, sig8, h0, Mag, gamma, sigv = theta
        theta = [
            self.provider.get_param("Omega_Lambda"), # NOTE: this needs to be a derived parameter
            self.provider.get_param("Omega_m"),
            self.provider.get_param("Omega_b"),
            self.provider.get_param("sigma8"),
            h,
            self.provider.get_param("gamma"),
            self.provider.get_param("Mag"),
            self.provider.get_param("sigv")
        ]
        cov_tot = self.cov_pan_fil + cov_PV_mod_8(theta, self.zvec_CMB, self.rdirs, lum_dis, nmod)

        if np.any(np.linalg.eigvals(cov_tot) < 0): return -np.inf
        
        inv_cov_tot = np.linalg.inv(cov_tot)
        signo, log_det_tot = np.linalg.slogdet(cov_tot)
        inv_L_tot_fil = np.linalg.cholesky(inv_cov_tot)
        ####################

        mahalanobis = np.linalg.multi_dot([vecdif,inv_L_tot_fil,inv_L_tot_fil.T,vecdif])
        return -0.5 * mahalanobis - 0.5 * log_det_tot


def Pantheon_data():
    """
        Extracts information from Pantheon data
    """
    dtype = {
    'names': ('CID', 'IDSURVEY', 'zHD', 'zHDERR', 'zCMB', 'zCMBERR', 'zHEL',
              'zHELERR', 'm_b_corr', 'm_b_corr_err_DIAG', 'MU_SH0ES',
              'MU_SH0ES_ERR_DIAG', 'CEPH_DIST', 'IS_CALIBRATOR',
              'USED_IN_SH0ES_HF', 'c', 'cERR', 'x1', 'x1ERR', 'mB', 'mBERR',
              'x0', 'x0ERR', 'COV_x1_c', 'COV_x1_x0', 'COV_c_x0', 'RA', 'DEC',
              'HOST_RA', 'HOST_DEC', 'HOST_ANGSEP', 'VPEC', 'VPECERR', 'MWEBV',
              'HOST_LOGMASS', 'HOST_LOGMASS_ERR', 'PKMJD', 'PKMJDERR', 'NDOF',
              'FITCHI2', 'FITPROB', 'm_b_corr_err_RAW', 'm_b_corr_err_VPEC',
              'biasCor_m_b', 'biasCorErr_m_b', 'biasCor_m_b_COVSCALE',
              'biasCor_m_b_COVADD'),
    'formats': ['U10'] + ['f8']*46  # 'U10' for the first 'string' column, 'f8' for the numeric columns
    }

    # file_path = "/share/storage3/bets/camilo_storage3/Pantheon/Pantheon+SH0ES.dat" # TODO: maybe point to pp.dat
    file_path = "/home/joaoreboucas/cocoa/Cocoa/cobaya/cobaya/likelihoods/mylike/data/pp.dat"
    data = np.genfromtxt(file_path, skip_header=1, dtype=dtype)

    # file_path = "/share/storage3/bets/camilo_storage3/Pantheon/Pantheon+SH0ES_STAT+SYS.cov" # TODO: maybe point to pp.cov
    file_path = "/home/joaoreboucas/cocoa/Cocoa/cobaya/cobaya/likelihoods/mylike/data/pp.cov"
    cov_vec = np.loadtxt(file_path, delimiter=' ') 
    matrix_size = int(np.sqrt(len(cov_vec[1:])))
    cov_matrix = np.reshape(cov_vec[1:], (matrix_size, matrix_size))

    zvec = data['zCMB'].T
    zHEL = data['zHEL'].T
    mvec = data['m_b_corr'].T
    RA = data['DEC'].T
    DEC = data['RA'].T

    indxs = np.where(zvec>=0.01)
    zvec_CMB = zvec[indxs]
    mvecc = mvec[indxs]
    zvec_HEL = zHEL[indxs]

    RA_vecc = RA[indxs]
    DEC_vecc = DEC[indxs]

    cov_matrix_fil = cov_matrix[indxs[0],:]
    cov_matrix_fil = cov_matrix_fil[:,indxs[0]]  # Eliminar columnas
    cov_pan_fil = cov_matrix_fil

    RA_rad = RA_vecc*np.pi/180
    DEC_rad = DEC_vecc*np.pi/180
    rdirs = []
    for i in range(len(zvec_CMB)):
        RAi = RA_rad[i] ; DECi = DEC_rad[i]
        diri = np.array([np.sin(RAi)*np.cos(DECi),np.sin(RAi)*np.sin(DECi),np.cos(RAi)])
        rdirs.append(diri)
    return mvecc, zvec_CMB, zvec_HEL, rdirs, cov_pan_fil

def Hub(h0,Om, OL,z):
    Ok = 1 - Om - OL
    return h0*100*np.sqrt( Om*(1+z)**3 + OL + Ok*(1+z)*(1+z) )/c

def K_par(x):
    return np.sin(x)/x - 2 * (np.sin(x)/x/x - np.cos(x)/x) /x
def K_per(x):
    return (np.sin(x)/x/x - np.cos(x)/x)/x

def f(h0,Om, OL,gamma,z):
    Omz = Om*(1+z)**3 *(h0*100/c/Hub(h0,Om, OL,z))**2
    return ( Omz )**(gamma)

# Define the function to calculate the growth factor and its derivative
def get_growth_function_derivative(results, z_values):
    # Extract the growth factor G(z) from CAMB
    growth_factor = results.get_redshift_evolution(1, z_values, ['delta_tot'])[:, 0]
    # Normalize the growth factor (G(z=0) = 1)
    growth_factor /= growth_factor[0]
    G_interp = InterpolatedUnivariateSpline(z_values, growth_factor, k=3)
    return G_interp#, G_der_tau

def cov_PV_mod_8(theta, zvecc, rdirs, lumdis, nmod):
    Nsns = len(lumdis)
    OL, Om, Ob, sig8, h0, Mag, gamma, sigv = theta
    Oc = Om - Ob
    Ok = 1-Om-OL
    #################### 
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=100*h0, ombh2=Ob*h0*h0, omch2=Oc*h0*h0, omk=Ok)
    pars.InitPower.set_params(As=2e-9, ns=ns)
    pars.set_matter_power(redshifts=[0], kmax=10.0)
    results = camb.get_results(pars)

    # Get sigma_8 and calculate the right pk later
    sigma8 = results.get_sigma8()[0] # TODO: add to requirements
    As= 2e-9*(sig8/sigma8)**2

    # Extract the growth factor G(z) from CAMB and calculate G'
    redshifts = np.linspace(0,3,100)
    growth_factor = results.get_redshift_evolution(1, redshifts, ['delta_tot'])[:, 0]
    growth_factor /= growth_factor[0] # TODO: add to requirements
    growth_function_interp = interp1d(redshifts, growth_factor, kind='cubic', fill_value="extrapolate")
    ####################

    ############
    kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=10, npoints = 200)
    pk = pk*As/2e-9 # TODO: add to requirements
    ############

    ### implementacao da matriz
    ctest = np.zeros((Nsns,Nsns), dtype=np.float64)
    constant = 5/np.log(10)
    ludis = lumdis/h0/100*c
    twopi2 = 2*np.pi**2
    nonlinpart = (sigv/c)**2 - (240/c)**2
    rs = results.comoving_radial_distance(zvecc) # TODO: add to requirements
    Gfactor = - f(h0,Om,OL,gamma,zvecc)*growth_function_interp(zvecc)/(1+zvecc)

        ############## Here goes the factor that considers curvature
    if Ok > 0: # closed universe
        R0 = c/(100*h0*np.sqrt(Ok))
        ckvec = np.cos(rs/R0)
    elif Ok<0 : # open universe
        R0 = c/(100*h0*np.sqrt(-Ok))
        ckvec = np.cosh(rs/R0)
    else : ckvec = np.ones(Nsns) # flat universe
        ##############

    for i in range(nmod):
        zi = zvecc[i]# ; RAi = RA_fil_rad[i] ; DECi = DEC_fil_rad[i]
        ri = rs[i] ; ri_dir = rdirs[i]

        for j in range(i, nmod):
            zj = zvecc[j]# ; RAj = RA_fil_rad[j] ; DECj = DEC_fil_rad[j]
            rj = rs[j] ; rj_dir = rdirs[j]

            rij = ri*ri_dir-rj*rj_dir
            r = np.sqrt(rij@rij) # in Mpc too

            ci = constant*(1-(1+zi)**2 /Hub(h0,Om,OL,zi)/ludis[i]*ckvec[i])
            cj = constant*(1-(1+zj)**2 /Hub(h0,Om,OL,zj)/ludis[j]*ckvec[j])

            G_der_tau_i = Gfactor[i]*Hub(h0,Om,OL,zi)
            G_der_tau_j = Gfactor[j]*Hub(h0,Om,OL,zj)

            #################
            if r == 0:
                integrand = pk[0, :] /3 / twopi2
                e_corrij = trapezoid(integrand, kh)
                nonlin = nonlinpart
            else:
                cosi = ri_dir@rij/r ; sini = np.sqrt(1-cosi**2)
                cosj = rj_dir@rij/r ; sinj = np.sqrt(1-cosj**2)

                integrand_par = pk[0, :] * K_par(kh*r/h0) / twopi2
                integrand_per = pk[0, :] * K_per(kh*r/h0) / twopi2
                epar = trapezoid(integrand_par, kh)
                eper = trapezoid(integrand_per, kh)
                e_corrij = sini*sinj*eper + cosi*cosj*epar
                nonlin = 0            

            ctest[i][j] = ci*cj*e_corrij*G_der_tau_i*G_der_tau_j + nonlin*ci*cj
            if i != j:
                ctest[j][i] = ctest[i][j]
    return ctest

#####################


def logprior(theta):
    OL, Om, Ob, sig8, h0, Mag, gamma, sigv = theta
    Ok = 1-Om-OL
    #################### priors
    if not (4*Ok**3 + 27*OL*Om**2 > 0): return -np.inf

    if not (-0.6<Ok<0.6)    : return -np.inf
    if not (0<OL<1.5)       : return -np.inf
    if not (Ob<Om<1.5)       : return -np.inf
    if not (0.005<Ob<0.2)   : return -np.inf
    if not (0<sig8<2)       : return -np.inf
    if not (0.4<h0<1)       : return -np.inf
    if not (-21<Mag<-18)    : return -np.inf
    if not (-1<gamma<3)     : return -np.inf
    if not (125<sigv<325)   : return -np.inf

    #### prior abs Mag
    Magmean = -19.253 ; sigMag = 0.029
    priorMag =  ((Mag-Magmean)/sigMag)**2
    #### prior BBN
    prior_bbn = ((Ob*h0*h0 - 0.02196) / 0.00063)**2
    return -0.5 * priorMag - 0.5 * prior_bbn

