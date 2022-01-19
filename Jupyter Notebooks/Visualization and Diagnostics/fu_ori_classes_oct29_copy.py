# 11 October 2019
# FU Ori Modeling Classes

""" 
This .py file stores all of the necessary classes for FU Ori User Interface
to make it easier to keep track of everything. Hopefully.
"""
# Imports
import numpy as np
from scipy import optimize
from scipy import interpolate
from scipy import integrate
from astropy.convolution import convolve, convolve_fft
import matplotlib.pyplot as plt

from fu_ori_functions_oct29_copy import *

# Constants
G = 6.67259e-8
SIG_SB = 5.67051e-5
M_SUN = 1.99e33
R_SUN = 6.96e10
L_SUN = 3.839e33
h_PLANCK = 6.6260755e-27
c_LIGHT = 2.99792458e10
k_BOLTZ = 1.380658e-16
sec_YEAR = 365*24*60*60

RAD_MAX_DISK = 1.361
ATMOS_FACTOR = 100

ISM_AVG = 3.1

class Annulus:
    
    # Initializer / Instance Attributes
    def __init__(self, temp, grav, r_inner, r_outer, directory):
        self.temp = temp
        self.grav = grav
        self.r_inner = r_inner
        self.r_outer = r_outer
        self.directory = directory
        
        self.area = 2*np.pi*(r_outer**2 - r_inner**2)
        
        # Non-interpolated quantities
        self.lums = 0
        self.wavelengths = 0

    # instance methods
    
    # Setting the luminosity
    def setSpectrumBlackbody(self, waves):
        self.wavelengths, self.lums = getBlackbody(waves, self.temp)
        self.wavelengths = self.wavelengths*1e8

    def setSpectrumFromDirec(self, spec_model, template_temp, template_grav, atm_table):
        if spec_model == 'stellar atmospheres':
            try:
                self.wavelengths, self.lums = getLumFromDirec(self.temp, self.grav, self.directory, spec_model, atm_table)
#                print('Used stellar atmosphere at T = ' + str(int(self.temp)) + '.')
            except:
#                print('No stellar atmosphere for annulus of T = ' + str(int(self.temp)) + '. Used blackbody.')
                direc_waves = getLumFromDirec(template_temp, template_grav, self.directory, spec_model, atm_table)[0]
                self.wavelengths = getBlackbody(direc_waves*1e-8, self.temp)[0]/1e-8
                self.lums = getBlackbody(direc_waves*1e-8, self.temp)[1]
        elif spec_model == 'blackbodies':
            direc_waves = getLumFromDirec(template_temp, template_grav, self.directory, 'stellar atmospheres', atm_table)[0]
            self.wavelengths = getBlackbody(direc_waves*1e-8, self.temp)[0]/1e-8
            self.lums = getBlackbody(direc_waves*1e-8, self.temp)[1]
    
    # Just plotting luminosity (**not multiplied by area**)
    def getLuminosity(self, wave_lower, wave_upper, data_type):
        if data_type == 'non-interpolated':
            waves, lums = self.wavelengths, self.lums
        else:
            pass
#            print('Data type does not exist.')

        ind_lower = np.searchsorted(waves, wave_lower)
        ind_upper = np.searchsorted(waves, wave_upper)
        return (waves[ind_lower:ind_upper], lums[ind_lower:ind_upper])

class FUOri:

    # Initializer / Instance Attributes
    def __init__(self, r_star, r_inner, r_outer, m_star, m_dot, inc, a_v, atm_table, pl_index):
        self.r_star = r_star
        self.r_inner = r_inner
        self.r_outer= r_outer
        self.m_star = m_star
        self.m_dot = m_dot
        self.inc = inc
        self.a_v = a_v
        self.pl_index = pl_index
        
        self.atm_table = atm_table
        
        self.area_tot = 2*np.pi*(r_outer**2 - r_inner**2)
        
        # Storing annuli
        self.temps = 0
        self.annuli = []
        self.r_a, self.r_b = 0, 0
        
        # Storing model spectrum
        self.mod_spec_waves = 0
        self.mod_spec_lums = 0
        
    # instance methods
    
    # Setting radii and temperatures of annuli based on desired range
    def setAnnuliValues(self, temp_max_poss, temp_min_poss, temp_binning, r_binning_outer):
        self.temps, self.r_a, self.r_b = generateMasterList(self.r_inner, self.r_outer, self.m_dot, \
                          self.m_star, temp_max_poss, temp_min_poss, temp_binning, r_binning_outer, self.pl_index)
    
    # Creating each annulus
    def createAnnuliFromDirec(self, grav, directory, model):
        for i in range(len(self.temps)):
            annulus = Annulus(self.temps[i], grav, self.r_a[i], self.r_b[i], directory)
            annulus.setSpectrumFromDirec(model, self.temps[0], grav, self.atm_table)
            self.annuli.append(annulus)
            
    # Making a model spectrum:
    
    # Weighting by area is done here !!!
    def createModelSpectrum(self, model, redden):
        # For a non-interpolated (stellar atmosphere) created disk:
        if model == 'stellar atmospheres':
            added_lums = np.zeros(len(self.annuli[0].lums))
            for i in range(len(self.annuli)):
                added_lums += self.annuli[i].lums*self.annuli[i].area
                
            self.mod_spec_waves = self.annuli[0].wavelengths
            self.mod_spec_lums = added_lums
        if redden:
            a_lambda = self.a_v*makeExtinctionCurve(self.mod_spec_waves, ISM_AVG)[1]
            self.mod_spec_lums = self.mod_spec_lums*10**(-a_lambda/2.5)

            
     
        
    # Just plotting the temperature profile used in modeling
    def getTempProfile(self):
        r_avg = np.mean((self.r_b, self.r_a), axis=0)
        return r_avg, self.temps
    
    
    # Analytically getting the total luminosity
    def getTotalLum(self):
        tot = 0
        for i in range(len(self.annuli)):
            tot += SIG_SB*(self.annuli[i].temp**4)*self.annuli[i].area
        return tot
    
class Star:
    
    # Initializer / Instance Attributes
    def __init__(self, temp, grav, m_star, r_star, directory, inc, a_v, atm_table):
        self.temp = temp
        self.grav = grav
        self.m_star = m_star
        self.r_star = r_star
        self.directory = directory
        self.inc = inc
        self.a_v = a_v
        
        self.atm_table = atm_table
        
        self.area = 4*np.pi*r_star**2
        
        # Non-interpolated quantities
        self.lums = 0
        self.wavelengths = 0
        
        # Interpolated quantities
        self.lums_interp = 0
        self.wavelengths_interp = 0
        self.lums_broad = 0


    # instance methods
    
    # Setting the luminosity
    def setSpectrumBlackbody(self, waves):
        self.wavelengths, self.lums = getBlackbody(waves, self.temp)
        self.wavelengths = self.wavelengths*1e8

    def setSpectrumFromDirec(self, spec_model, template_temp, template_grav, atm_table):
        if spec_model == 'stellar atmospheres':
            try:
                self.wavelengths, self.lums = getLumFromDirec(self.temp, self.grav, self.directory, spec_model, atm_table)
#                print('Used stellar atmosphere at T = ' + str(int(self.temp)) + '.')
            except:
#                print('No stellar atmosphere for annulus of T = ' + str(int(self.temp)) + '. Used blackbody.')
                direc_waves = getLumFromDirec(template_temp, template_grav, self.directory, spec_model, atm_table)[0]
                self.wavelengths = getBlackbody(direc_waves*1e-8, self.temp)[0]/1e-8
                self.lums = getBlackbody(direc_waves*1e-8, self.temp)[1]
        else:
            print('Model not available. Choose another or use blackbody setting.')
            
    
    # Just plotting luminosity (**not multiplied by area**)
    def getLuminosity(self, wave_lower, wave_upper, redden):
        waves, lums = self.wavelengths, self.lums

        if redden:
            a_lambda = self.a_v*makeExtinctionCurve(waves, ISM_AVG)[1]
            lums = lums*10**(-a_lambda/2.5)            
        ind_lower = np.searchsorted(waves, wave_lower)
        ind_upper = np.searchsorted(waves, wave_upper)
        return (waves[ind_lower:ind_upper], lums[ind_lower:ind_upper])

    # Broadening
    def broadenStellarSpectrum(self):
        waves = self.wavelengths
        lums = self.lums

        rot_profile = stellarRotationProfileNorm(self.m_star, waves, np.mean(waves), self.r_star, self.inc)
        lums_broad = convolve(lums, rot_profile)
        self.lums_broad = lums_broad
    

    
    