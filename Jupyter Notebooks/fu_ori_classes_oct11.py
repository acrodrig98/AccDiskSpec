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

from fu_ori_functions_oct11 import *

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
                print('Used stellar atmosphere at T = ' + str(int(self.temp)) + '.')
            except:
                print('No stellar atmosphere for annulus of T = ' + str(int(self.temp)) + '. Used blackbody.')
                direc_waves = getLumFromDirec(template_temp, template_grav, self.directory, spec_model, atm_table)[0]
                self.wavelengths = getBlackbody(direc_waves*1e-8, self.temp)[0]/1e-8
                self.lums = getBlackbody(direc_waves*1e-8, self.temp)[1]
        else:
            print('Model not available. Choose another or use blackbody setting.')
    
    # Just plotting luminosity (**not multiplied by area**)
    def getLuminosity(self, wave_lower, wave_upper, data_type):
        if data_type == 'non-interpolated':
            waves, lums = self.wavelengths, self.lums
        elif data_type == 'interpolated':
            waves, lums = self.wavelengths_interp, self.lums_interp
        elif data_type == 'broadened':       
            waves, lums = self.wavelengths_interp, self.lums_broad
        else:
            print('Data type does not exist.')
            
        ind_lower = np.searchsorted(waves, wave_lower)
        ind_upper = np.searchsorted(waves, wave_upper)
        return (waves[ind_lower:ind_upper], lums[ind_lower:ind_upper])

    
    # Broadening
    # Broadening
    def broadenSpectrum(self, m_star, inc):
        waves = self.wavelengths_interp
        lums = self.lums_interp
        r_avg = np.mean([self.r_outer, self.r_inner])
        vel = getObservedVelocity(m_star, r_avg, inc)
        
        # broad_profile = rotationProfileNorm(m_star, waves, np.mean(waves), r_avg, inc)
        # lums_broad = convolve(lums, broad_profile)
        
        # Converting to km/s
        self.lums_broad = rotBroad(waves, lums, 1e-5*vel)
        # self.lums_broad = lums_broad
        
class FUOri:

    # Initializer / Instance Attributes
    def __init__(self, r_star, r_inner, r_outer, m_star, m_dot, inc, a_v, atm_table):
        self.r_star = r_star
        self.r_inner = r_inner
        self.r_outer= r_outer
        self.m_star = m_star
        self.m_dot = m_dot
        self.inc = inc
        self.a_v = a_v
        
        self.atm_table = atm_table
        
        self.area_tot = 2*np.pi*(r_outer**2 - r_inner**2)
        
        # Storing annuli
        self.temps = 0
        self.annuli = []
        self.r_a, self.r_b = 0, 0
        
        # Storing model spectrum
        self.mod_spec_waves = 0
        self.mod_spec_lums_broad = 0
        self.mod_spec_lums = 0
        
    # instance methods
    
    # Setting radii and temperatures of annuli based on desired range
    def setAnnuliValues(self, temp_max_poss, temp_min_poss, temp_binning, r_binning_outer):
        self.temps, self.r_a, self.r_b = generateMasterList(self.r_inner, self.r_outer, self.m_dot, \
                          self.m_star, temp_max_poss, temp_min_poss, temp_binning, r_binning_outer)
    
    # Creating each annulus
    def createAnnuliFromDirec(self, grav, directory, model):
        for i in range(len(self.temps)):
            annulus = Annulus(self.temps[i], grav, self.r_a[i], self.r_b[i], directory)
            annulus.setSpectrumFromDirec(model, self.temps[0], grav, self.atm_table)
            self.annuli.append(annulus)
            
    def createAnnuliBlackbody(self, waves):
        waves = waves*1e-8
        for i in range(len(self.temps)):
            annulus = Annulus(self.temps[i], None, self.r_a[i], self.r_b[i], None)
            annulus.setSpectrumBlackbody(waves)
            self.annuli.append(annulus)
             
    # Making a model spectrum:
    # 1. Prepare annuli
    def prepareAnnuli(self, wave_lower, wave_upper, binning, broaden, interp_type):
        # Feeding in uninterpolated quantities
        for i in range(len(self.annuli)):
            waves = self.annuli[i].wavelengths
            lums = self.annuli[i].lums
            # Assigning interpolated quantities
            self.annuli[i].wavelengths_interp, self.annuli[i].lums_interp=\
            prepareAnnulus(waves, lums, wave_lower, wave_upper, binning, interp_type)
            
            # Broaden
            if broaden:
                self.annuli[i].broadenSpectrum(self.m_star, self.inc)
        print_state = 'Annuli prepared from ' + str(int(wave_lower)) + '-'+str(int(wave_upper))\
              + ' Angstrom with binning of ' + str(binning) + ' Angstrom.'    
        if broaden:
            print(print_state + ' Broadening implemented.')
        else:
            print(print_state)
            
    
    # 2. Choose to implement broadening or not.
    # ** Weighting by area is done here!! **
    def createModelSpectrum(self, model, broaden):
        # For an interpolated (stellar atmosphere) created disk:
        if model == 'stellar atmospheres':
            added_lums = np.zeros(len(self.annuli[0].lums_interp))
            added_lums_broad = np.zeros(len(self.annuli[0].lums_interp))
            for i in range(len(self.annuli)):
                if broaden:
                    added_lums_broad += self.annuli[i].lums_broad*self.annuli[i].area
                added_lums += self.annuli[i].lums_interp*self.annuli[i].area
                
            self.mod_spec_waves = self.annuli[0].wavelengths_interp
            self.mod_spec_lums = added_lums
            self.mod_spec_lums_broad = added_lums_broad
        # For a pure blackbody disk:
        else:
            added_lums = np.zeros(len(self.annuli[0].lums))
            for i in range(len(self.annuli)):
                if broaden:
                    print('Broadening not available for blackbodies.')
                    return
                added_lums += self.annuli[i].lums*self.annuli[i].area
            self.mod_spec_waves = self.annuli[0].wavelengths
            self.mod_spec_lums = added_lums
            
    def viewModelSpectrum(self, wave_lower, wave_upper, broad, redden):
        waves = self.mod_spec_waves
        if broad:
            lums = self.mod_spec_lums_broad
        else:
            lums = self.mod_spec_lums
        if redden:
            a_lambda = self.a_v*makeExtinctionCurve(waves, ISM_AVG)[1]
            lums = lums*10**(-a_lambda/2.5)
        ind_lower = np.searchsorted(waves, wave_lower)
        ind_upper = np.searchsorted(waves, wave_upper)
        return (waves[ind_lower:ind_upper], lums[ind_lower:ind_upper])        
        
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
    
    # Calculating the fractional fluxes/luminosities as function of temp/dist
    def calculateFracLums(self, wave_lower, wave_upper, data_type, x_axis_type):
        frac_lums = np.zeros(len(self.annuli))
        temps = np.zeros(len(self.annuli))
        temps_diff = np.zeros(len(self.annuli))
        r_avg = np.zeros(len(self.annuli))
        total_lum = self.getTotalLum()

        for i in range(len(self.annuli)):
            annulus = self.annuli[i]
            x = annulus.getLuminosity(wave_lower, wave_upper, data_type)[0]
            y = annulus.getLuminosity(wave_lower, wave_upper, data_type)[1]*annulus.area
            frac_lums[i] = integrate.simps(y,x)*1e-8
            temps[i] = annulus.temp
            r_avg[i] = (annulus.r_inner + annulus.r_outer)/2
            temps_diff[i] = tempFUOriDisk([annulus.r_inner], self.r_star, self.m_dot,\
                      self.m_star) \
            - tempFUOriDisk([annulus.r_outer], self.r_star, self.m_dot, self.m_star)
            
        # Inner 
        inner_disk_diff = np.abs(self.annuli[1].temp - self.annuli[0].temp)
        temps_diff[0] = inner_disk_diff
        if x_axis_type == 'distance':
            return r_avg/self.r_star, (frac_lums/total_lum)*(inner_disk_diff/temps_diff)
        if x_axis_type == 'temperature':
            return temps, (frac_lums/total_lum)*(inner_disk_diff/temps_diff)
    
    
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

    def setSpectrumFromDirec(self, spec_model, template_temp, template_grav):
        if spec_model == 'stellar atmospheres':
            try:
                self.wavelengths, self.lums = getLumFromDirec(self.temp, self.grav, self.directory, spec_model, self.atm_table)
                print('Used stellar atmosphere at T = ' + str(int(self.temp)) + '.')
            except:
                print('No stellar atmosphere for annulus of T = ' + str(int(self.temp)) + '. Used blackbody.')
                direc_waves = getLumFromDirec(template_temp, template_grav, self.directory, spec_model, self.atm_table)[0]
                self.wavelengths = getBlackbody(direc_waves*1e-8, self.temp)[0]/1e-8
                self.lums = getBlackbody(direc_waves*1e-8, self.temp)[1]
        else:
            print('Model not available. Choose another or use blackbody setting.')
            
    def prepareSpectrum(self, wave_lower, wave_upper, binning, interp_type):
        self.wavelengths_interp, self.lums_interp=\
            prepareAnnulus(self.wavelengths, self.lums, wave_lower, wave_upper, binning, interp_type)
    
    # Just plotting luminosity (**not multiplied by area**)
    def getLuminosity(self, wave_lower, wave_upper, data_type, redden):
        if data_type == 'non-interpolated':
            waves, lums = self.wavelengths, self.lums
        elif data_type == 'interpolated':
            waves, lums = self.wavelengths_interp, self.lums_interp
        elif data_type == 'broadened':       
            waves, lums = self.wavelengths_interp, self.lums_broad
        else:
            print('Data type does not exist.')
        if redden:
            a_lambda = self.a_v*makeExtinctionCurve(waves, ISM_AVG)[1]
            lums = lums*10**(-a_lambda/2.5)            
        ind_lower = np.searchsorted(waves, wave_lower)
        ind_upper = np.searchsorted(waves, wave_upper)
        return (waves[ind_lower:ind_upper], lums[ind_lower:ind_upper])

    # Broadening
    def broadenStellarSpectrum(self):
        waves = self.wavelengths_interp
        lums = self.lums_interp

        rot_profile = stellarRotationProfileNorm(self.m_star, waves, np.mean(waves), self.r_star, self.inc)
        lums_broad = convolve(lums, rot_profile)
        self.lums_broad = lums_broad
        
        
class DiskAtmosphere:
    
    # Initializer / Instance Attributes
    def __init__(self, temp, grav, vsini, directory, atm_table):
        self.temp = temp
        self.grav = grav
        self.directory = directory
        self.vsini = vsini
        
        self.atm_table = atm_table
        
        # Non-interpolated quantities
        self.lums = 0
        self.wavelengths = 0
        
        # Interpolated quantities
        self.lums_interp = 0
        self.wavelengths_interp = 0
        self.lums_broad = 0


    # instance methods
    
    # Setting the luminosity

    def setSpectrumFromDirec(self, spec_model, template_temp, template_grav):
        if spec_model == 'stellar atmospheres':
            try:
                self.wavelengths, self.lums = getLumFromDirec(self.temp, self.grav, self.directory, spec_model, self.atm_table)
                print('Used stellar atmosphere at T = ' + str(int(self.temp)) + '.')
            except:
                print('No stellar atmosphere for annulus of T = ' + str(int(self.temp)) + '. Used blackbody.')
                direc_waves = getLumFromDirec(template_temp, template_grav, self.directory, spec_model, self.atm_table)[0]
                self.wavelengths = getBlackbody(direc_waves*1e-8, self.temp)[0]/1e-8
                self.lums = getBlackbody(direc_waves*1e-8, self.temp)[1]
        else:
            print('Model not available. Choose another or use blackbody setting.')
            
    def prepareSpectrum(self, wave_lower, wave_upper, binning, interp_type):
        self.wavelengths_interp, self.lums_interp=\
            prepareAnnulus(self.wavelengths, self.lums, wave_lower, wave_upper, binning, interp_type)
 

    # Broadening
    def broadenDiskAtmSpectrum(self):
        waves = self.wavelengths_interp
        lums = self.lums_interp
        vel = self.vsini
#        self.lums_broad = rotBroad(waves, lums, vel)
        
        # broad_profile = rotationProfileNormDisk(waves, np.mean(waves), vel)
        # lums_broad = convolve(lums, broad_profile)
        
        # Converting to km/s
        self.lums_broad = rotBroad(waves, lums, 1e-5*vel)
        # self.lums_broad = lums_broad
