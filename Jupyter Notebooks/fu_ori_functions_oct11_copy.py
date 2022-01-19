# 11 October 2019
# FU Ori Modeling Functions

""" 
This .py file stores all of the necessary functions for FU Ori Full Modeling
to make it easier to keep track of everything. Hopefully.
"""
# Imports
from __future__ import print_function, division
import numpy as np
from PyAstronomy.pyaC import pyaErrors as PE
import six.moves as smo
import numpy as np
from scipy import optimize
from scipy import interpolate
from astropy.convolution import convolve, convolve_fft
import matplotlib.pyplot as plt
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

# Temperature functions
def tempKepDisk(r, r_inner, m_dot, m_star):
    term1 = 3*G*m_star*m_dot / (8 * np.pi * SIG_SB * (r**3))
    term2 = (1 - (r_inner/r)**(1/2))
    return (term1 * term2)**(1/4)

def tempFUOriDisk(r, r_inner, m_dot, m_star, pl_index):
    res = np.zeros(len(r))
    for i in range(len(r)):
        if r[i] <= RAD_MAX_DISK*r_inner:
            t_max =  tempKepDisk(RAD_MAX_DISK*r_inner, r_inner, m_dot, m_star)
            res[i] = t_max*((r[i]/(RAD_MAX_DISK*r_inner))**pl_index)
        else:
            res[i] = tempKepDisk(r[i], r_inner, m_dot, m_star)
    return res

def tempFUOriDiskMod(r, r_inner, m_dot, m_star, pl_index):
    if r <= RAD_MAX_DISK*r_inner:
        t_max = tempKepDisk(RAD_MAX_DISK*r_inner, r_inner, m_dot, m_star)
        res = t_max*((r/(RAD_MAX_DISK*r_inner))**pl_index)
    else:
        res = tempKepDisk(r, r_inner, m_dot, m_star)
    return res
    
def tempFUOriDiskMin(r, r_inner, m_dot, m_star, val, pl_index):
    return tempFUOriDiskMod(r, r_inner, m_dot, m_star, pl_index) - val

# Annuli-generating functions
    
def find_nearest(array, value, side):
    array = np.asarray(array)
    min_vals = array-value
    max_vals = -min_vals
    if side == 'above':
        for i in range(len(min_vals)):
            if min_vals[i] < 0: min_vals[i] = np.inf
        idx = min_vals.argmin()
    if side == 'below':
        for i in range(len(min_vals)):
            if max_vals[i] < 0: max_vals[i] = np.inf
        idx = max_vals.argmin()
    return array[idx]

def getAvgOfPairs(arr):
    out_arr = np.zeros(len(arr)-1)
    for i in range(len(arr)-1):
        out_arr[i] = np.mean([arr[i], arr[i+1]])
    return out_arr

def makeOuterAnnuli(r_inner, r_outer, m_dot, m_star, r_start, r_binning, pl_index):
    r_list = np.arange(r_start, r_outer+r_binning, r_binning)
    r_a = r_list[:-1]
    r_b = r_list[1:]
    r_avg = np.mean((r_a,r_b), axis=0)
    temps = tempFUOriDisk(r_avg, r_inner, m_dot, m_star, pl_index)
    return temps, r_a, r_b

def generateMasterListInner(r_inner, r_tmax, m_dot, m_star, temp_max_poss, temp_min_poss, temp_binning, pl_index):
    # Max and min temperatures of defined disk
    temp_inside = tempFUOriDiskMod(r_inner, r_inner, m_dot, m_star, pl_index)
    temp_outside = tempFUOriDiskMod(r_tmax, r_inner, m_dot, m_star, pl_index)
    
    # These designations account for the inner temp being 
    # similar to the outer temp
    epsilon = temp_binning
    temp_in_greater = 0
    if temp_inside > temp_outside + epsilon:
        temp_in_greater = 1
    elif temp_outside > temp_inside + epsilon:
        temp_in_greater = -1
        
    # Looking at all possible temperatures of given library
    temp_prelim = np.arange(temp_min_poss - 0.5*temp_binning, temp_max_poss + 1.5*temp_binning, temp_binning)
    
    if temp_in_greater == 1:
        max_temp = tempFUOriDiskMod(r_inner, r_inner, m_dot, m_star, pl_index)
        min_temp = tempFUOriDiskMod(r_tmax, r_inner, m_dot, m_star, pl_index)
    elif temp_in_greater == -1:
        min_temp = tempFUOriDiskMod(r_inner, r_inner, m_dot, m_star, pl_index)
        max_temp = tempFUOriDiskMod(r_tmax, r_inner, m_dot, m_star, pl_index)
    else:
        low_lim = find_nearest(temp_prelim, temp_inside, 'below')
        high_lim = find_nearest(temp_prelim, temp_inside, 'above')
        return np.array([np.mean((low_lim, high_lim))]), np.array([r_inner]), np.array([r_tmax])
                                                                             
    # Finding ranges within the full temperature library
    min_nearest = find_nearest(temp_prelim, min_temp, 'above')
    max_nearest = find_nearest(temp_prelim, max_temp, 'below')
    
    # Making new list for annuli
    temp_spaced = np.arange(min_nearest, max_nearest + temp_binning, temp_binning)

    r_a = np.zeros(len(temp_spaced))
    for i in range(len(temp_spaced)):
        sol = optimize.root_scalar(tempFUOriDiskMin,args=(r_inner, m_dot, m_star, temp_spaced[i], pl_index),\
                                   bracket=[r_inner, r_tmax], method='brentq')
        r_a[i] = sol.root
        
    if temp_in_greater == 1:
        r_a = np.concatenate((r_a, [r_inner]))[::-1]
        r_b = np.concatenate((r_a[1:], [r_tmax]))
                            
    elif temp_in_greater == -1:
        r_a = np.concatenate(([r_inner], r_a))
        r_b = np.concatenate((r_a[1:], [r_tmax]))
    
    # Average temperatures of annuli
    temp_radiating_prelim = getAvgOfPairs(temp_spaced)
    temp_radiating = np.concatenate(([np.min(temp_radiating_prelim) - temp_binning],\
                                     temp_radiating_prelim,\
                                     [np.max(temp_radiating_prelim) + temp_binning]))
    
    # Adding temperatures below final stellar atmosphere
    # EXCLUDING FINAL VALUES since they're accounted for in the outer annuli
    temp_radiating_curr = temp_radiating
    if temp_in_greater == 1:
        temp_radiating_curr = temp_radiating_curr[::-1]
    r_a_curr = r_a
    r_b_curr = r_b

    return temp_radiating_curr, r_a_curr, r_b_curr

def generateMasterListOuter(r_inner, r_outer, r_tmax, m_dot, m_star, temp_max_poss, temp_min_poss, temp_binning, r_binning_outer, pl_index):
    # Max and min temperatures of defined disk
    max_temp = tempFUOriDiskMod(r_tmax, r_inner, m_dot, m_star, pl_index)
    min_temp = tempFUOriDiskMod(r_outer, r_inner, m_dot, m_star, pl_index)
    
    # Looking at all possible temperatures of given library
    temp_prelim = np.arange(temp_min_poss - 0.5*temp_binning, temp_max_poss + 1.5*temp_binning, temp_binning)
    min_nearest = find_nearest(temp_prelim, min_temp, 'above')
    max_nearest = find_nearest(temp_prelim, max_temp, 'below')
    
    # Making new list for annuli
    temp_spaced = np.arange(min_nearest, max_nearest + temp_binning, temp_binning)
    r_a = np.zeros(len(temp_spaced)+1)
    for i in range(len(temp_spaced)):
        sol = optimize.root_scalar(tempFUOriDiskMin,args=(r_inner, m_dot, m_star, temp_spaced[i], pl_index),\
                                   bracket=[r_tmax, r_outer], method='brentq')
        r_a[i] = sol.root
    r_a[-1] = r_tmax
    r_b = np.concatenate(([r_outer], r_a))[:-1]
    
    # Average temperatures of annuli
    temp_radiating_prelim = getAvgOfPairs(temp_spaced)
    temp_radiating = np.concatenate(([np.min(temp_radiating_prelim) - temp_binning],\
                                     temp_radiating_prelim,\
                                     [np.max(temp_radiating_prelim) + temp_binning]))
    
    # Adding temperatures below final stellar atmosphere
    # EXCLUDING FINAL VALUES since they're accounted for in the outer annuli
    temp_radiating_curr = temp_radiating[::-1][:-1]
    r_a_curr = r_a[::-1][:-1]
    r_b_curr = r_b[::-1][:-1]
    r_start = r_b_curr[-1]
    temp_radiating_outer, r_a_outer, r_b_outer = makeOuterAnnuli(r_inner, r_outer,\
                                        m_dot, m_star, r_start, r_binning_outer, pl_index)
    
    temp_radiating_final = np.concatenate((temp_radiating_curr, temp_radiating_outer))
    r_a_final = np.concatenate((r_a_curr, r_a_outer))
    r_b_final = np.concatenate((r_b_curr, r_b_outer))
    
    return temp_radiating_final, r_a_final, r_b_final

def generateMasterList(r_inner, r_outer, m_dot, m_star, temp_max_poss, temp_min_poss, temp_binning, r_binning_outer, pl_index):
    # Replace this at some point?
    RAD_MAX_DIST = RAD_MAX_DISK*r_inner
    inner_lists = generateMasterListInner(r_inner, RAD_MAX_DIST,\
                                          m_dot, m_star, temp_max_poss, temp_min_poss, temp_binning, pl_index)
    outer_lists = generateMasterListOuter(r_inner, r_outer, RAD_MAX_DIST,\
                                          m_dot, m_star, temp_max_poss, temp_min_poss, temp_binning, r_binning_outer, pl_index)
    
    final_lists = []
    for i in range(len(inner_lists)):
        final_lists.append(np.concatenate((inner_lists[i], outer_lists[i])))
    return final_lists 

# Luminosity functions
    
def getBlackbody(wavelength, temp):
    term1 = 2*h_PLANCK*(c_LIGHT**2)*(wavelength**(-5))*np.pi
    term2 = (np.exp(h_PLANCK*c_LIGHT/(wavelength*k_BOLTZ*temp))-1)**(-1)
    return (wavelength, term1*term2)

def getLumFromDirecOLD(temp, grav, directory, model_type):
        atmos_file = directory + 'pr.lte' + str(int(temp/ATMOS_FACTOR)) \
                        + '-' + str(grav) + '-0.0.spec'
        txt = open(atmos_file)
        fulltxt = txt.readlines()
        # Cleaning lines
        newtxt = []
        for i in range(len(fulltxt)):
            line = fulltxt[i][:-1]
            line = line.split()
            newtxt.append(line)

        # Casting as floats
        newtxt = np.array(newtxt).astype(np.float64)
        
        # Choosing which model to use
        if model_type == 'stellar atmospheres':
            return (newtxt[:,0], newtxt[:,1])
        else:
            return (0,0)
        
def getLumFromDirec(temp, grav, directory, model_type, atm_table):
        
        # Choosing which model to use
        if model_type == 'stellar atmospheres':
            return atm_table[(temp, grav)][0], atm_table[(temp, grav)][1]
        else:
            return (0,0)
        
        
def direcToArray(directory):
    for temp in range(2000, 10000+ATMOS_FACTOR, ATMOS_FACTOR):
        for grav in [1.5, 4.0]:
            try:
                atmos_file = directory + 'pr.lte' + str(int(temp/ATMOS_FACTOR)) \
                        + '-' + str(grav) + '-0.0.spec'
                txt = open(atmos_file)
                fulltxt = txt.readlines()
                # Cleaning lines
                newtxt = []
                for i in range(len(fulltxt)):
                    line = fulltxt[i][:-1]
                    line = line.split()
                    newtxt.append(line)
        
                # Casting as floats
                newtxt = np.array(newtxt).astype(np.float64)
                
                # Choosing which model to use
                wav, lum = newtxt[:,0], newtxt[:,1]
                MASTER_ATMOS_TABLE[(temp, grav)] = np.array([wav, lum])
            except:
                print('No atmosphere found for temp = ' + str(temp) + \
                      ' grav = ' + str(grav))
        
# Model Spectrum functions
            
def prepareAnnulus(annuli_waves, annuli_lums, wave_lower, wave_upper, binning, interp_type):
    waves_binned = np.arange(wave_lower, wave_upper + binning, binning)
    
    if interp_type == 'linear' or 'cubic':
        waves, lum = annuli_waves, annuli_lums
        ind_lower = np.searchsorted(waves, wave_lower)
        ind_upper = np.searchsorted(waves, wave_upper)

        waves_trunc = waves[ind_lower-1:ind_upper+1]
        lum_trunc = lum[ind_lower-1:ind_upper+1]
        lum_interpolated = interpolate.interp1d(waves_trunc, lum_trunc, kind=interp_type)
        lum_binned = lum_interpolated(waves_binned)

        return waves_binned, lum_binned
    else:
        print('Interpolation type does not exist.')
        

# Line broadening kernels
def getObservedVelocity(m_star, r, inc):
    v = (G*m_star/r)**(1/2)
    return v*np.sin(inc)
        
        
def rotationProfileNorm(m_star, waves, wave_0, r, inc):
    waves = waves*1e-8
    wave_0 = wave_0*1e-8
    f_out = np.zeros(len(waves))
    obs_vel = getObservedVelocity(m_star, r, inc)
    wave_max = (wave_0*obs_vel/c_LIGHT)
    for i in range(len(waves)):
        if np.abs(waves[i]-wave_0) < wave_max:
            f_out[i] = (1/(np.pi*wave_max))*(1 - ((waves[i]-wave_0)/wave_max)**2)**(-1/2)
    return f_out 

def rotationProfileNormDisk(waves, wave_0, vsini):
    waves = waves*1e-8
    wave_0 = wave_0*1e-8
    f_out = np.zeros(len(waves))
    obs_vel = vsini
    wave_max = (wave_0*obs_vel/c_LIGHT)
    for i in range(len(waves)):
        if np.abs(waves[i]-wave_0) < wave_max:
            f_out[i] = (1/(np.pi*wave_max))*(1 - ((waves[i]-wave_0)/wave_max)**2)**(-1/2)
    return f_out 

def stellarRotationProfileNorm(m_star, waves, wave_0, r, inc):
    waves = waves*1e-8
    wave_0 = wave_0*1e-8
    f_out = np.zeros(len(waves))
    obs_vel = getObservedVelocity(m_star, r, inc)
    wave_max = (wave_0*obs_vel/c_LIGHT)
    for i in range(len(waves)):
        if np.abs(waves[i]-wave_0) < wave_max:
            f_out[i] = (1/(np.pi*wave_max))*np.exp(-((waves[i]-wave_0)/wave_max)**2)
    return f_out 

class _Gdl:
  
  def __init__(self, vsini):

    self.vc = vsini / 299792.458
  
  def gdl(self, dl, refwvl, dwl):
    self.dlmax = self.vc * refwvl
    result = np.zeros(len(dl))
    x = dl/self.dlmax
    indi = np.where(np.abs(x) < 1.0)[0]
    result[indi] = (1/(np.pi*self.dlmax))*(1. - (x[indi])**2)**(-0.5)
    result /= (np.sum(result) * dwl)
    return result
  

def rotBroad(wvl, flux, vsini, edgeHandling="firstlast"):

  # Check whether wavelength array is evenly spaced
  sp = wvl[1::] - wvl[0:-1]
  if abs(max(sp) - min(sp)) > 1e-6:
    raise(PE.PyAValError("Input wavelength array is not evenly spaced.",
                         where="pyasl.rotBroad",
                         solution="Use evenly spaced input array."))
  if vsini <= 0.0:
    raise(PE.PyAValError("vsini must be positive.", where="pyasl.rotBroad"))
  
  # Wavelength binsize
  dwl = wvl[1] - wvl[0]
  
  # Indices of the flux array to be returned
  validIndices = None
  
  if edgeHandling == "firstlast":
    # Number of bins additionally needed at the edges 
    binnu = int(np.floor(((vsini / 299792.458) * max(wvl)) / dwl)) + 1
    # Defined 'valid' indices to be returned
    validIndices = np.arange(len(flux)) + binnu
    # Adapt flux array
    front = np.ones(binnu) * flux[0]
    end = np.ones(binnu) * flux[-1]
    flux = np.concatenate( (front, flux, end) )
    # Adapt wavelength array
    front = (wvl[0] - (np.arange(binnu) + 1) * dwl)[::-1]
    end = wvl[-1] + (np.arange(binnu) + 1) * dwl
    wvl = np.concatenate( (front, wvl, end) )
  elif edgeHandling == "None":
    validIndices = np.arange(len(flux))
  else:
    raise(PE.PyAValError("Edge handling method '" + str(edgeHandling) + "' currently not supported.",
                         where="pyasl.rotBroad",
                         solution="Choose ones of the valid edge handling methods"))
    
  
  result = np.zeros(len(flux))
  gdl = _Gdl(vsini)
  
  for i in smo.range(len(flux)):
    dl = wvl[i] - wvl
    g = gdl.gdl(dl, wvl[i], dwl)
    result[i] = np.sum(flux * g)
  result *= dwl
  
  return result[validIndices]

# Extinction functions
# Calculation from Cardelli, Clayton, Mathis (1989)
def calcExtinction(x, r):
    if 0. <= x <= 1.1:
        return 0.574*(x**1.61) - (0.527*(x**1.61))/r
    elif 1.1 < x <= 3.3:
        y = x - 1.82
        return 1 + 0.17699*y - 0.50447*(y**2) - 0.02427*(y**3) + 0.72085*(y**4) \
        + 0.01979*(y**5) - 0.77530*(y**6) + 0.32999*(y**7)\
        +(1.41338*y + 2.28305*(y**2) + 1.07233*(y**3) - 5.38434*(y**4)\
        - 0.62251*(y**5) + 5.30260*(y**6) - 2.09002*(y**7) )/r
    elif 3.3 < x <= 8:
        if 5.9 <= x <= 8:
            f_a = -0.04473*(x - 5.9)**2 - 0.009779*(x - 5.9)**3
            f_b = 0.2130*(x - 5.9)**2 + 0.1207*(x - 5.9)**3
        else:
            f_a = 0
            f_b = 0
            
        return 1.752 - 0.316*x - 0.104/((x - 4.67)**2 + 0.341) \
                + f_a + (-3.090 + 1.825*x + 1.206/((x - 4.62)**2 + 0.263) + f_b)/r
    else:
        return 0
    
def makeExtinctionCurve(waves, r):    
    waves_mod = 1/(waves*1e-4)
    return waves, np.array([calcExtinction(wave, r) for wave in waves_mod])  

    
