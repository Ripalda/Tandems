# Random sampling of multijunction photovoltaic efficiencies. Jose M. Ripalda
# Requires doing "pip install json_tricks" before running
# Tested with Python 2.7 and 3.6
# SMARTS 2.9.5 is required only to generate a new set of random spectra.
# Provided files with ".npy" extension can be used instead of SMARTS to load a set of binned spectra.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time
import pdb
import json_tricks
import copy
import os.path
import subprocess as sub
from datetime import datetime
from scipy import integrate
import scipy.constants as con
hc = con.h*con.c
q = con.e

version = 1
print ('Tandems version',version)

np.set_printoptions(precision=3) # Print 3 decimal places only

colors = [(1, 0, 1), (0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]  # B -> G -> R
LGBT = LinearSegmentedColormap.from_list('LGBT', colors, 500)

# Load standard reference spectra ASTM G173
wavel, g1_5, d1_5 = np.loadtxt("AM1_5 smarts295.ext.txt", delimiter=',', usecols=(0,1,2), unpack=True)
Energies = 1e9*hc/q/wavel # wavel is wavelenght in nm
a1123456789 = [1, 1, 2, 3, 4, 5, 6, 7, 8, 9]

bindex = [] # "2D Array for spectral bin indexing. First index is number of bins, second is bin index. Returns a spectrum index for input in specs array.
bindex.append([0])
k = 1
for i in range(1, 10): #bin set
    bindex.append([])
    for j in range(0, i): #bin counter
        bindex[i].append(k)
        k += 1

# These arrays are used to speed up calculations by limiting the energy gap search space.
# Initial guess at where the eff maxima are.
# The search space is expanded as needed if high eff are found at the edges of the search space.
Emax = [] # Max energy for each gap
Emin = [] # Min energy for each gap
Emax.append([1.6]) # 1 junction
Emin.append([0.9])
Emax.append([1.40, 2.00]) # 2  junctions
Emin.append([0.85, 1.40])
Emax.append([1.20, 1.67, 2.00]) # 3  junctions
Emin.append([0.65, 1.15, 1.60])
Emax.append([1.10, 1.51, 1.90, 2.10]) # 4  junctions
Emin.append([0.50, 0.94, 1.25, 1.80])
Emax.append([0.95, 1.15, 1.55, 2.00, 2.25]) # 5  junctions
Emin.append([0.50, 0.83, 1.15, 1.50, 1.90])
Emax.append([0.95, 1.15, 1.45, 1.85, 2.10, 2.30]) # 6  junctions
Emin.append([0.50, 0.78, 1.05, 1.40, 1.70, 2.05])

# These arrays are used to discard band gap combinations that are too unevenly spaced in energy
# The search space is expanded as needed if high eff are found at the edges of the search space.
maxDif = []
minDif = []
maxDif.append([1.00]) # 2 junctions, max E difference between junctions
minDif.append([0.70]) # 2 junctions, min E difference between junctions
maxDif.append([0.65, 0.55])
minDif.append([0.50, 0.50]) # 3 junctions
maxDif.append([0.60, 0.55, 0.65])
minDif.append([0.45, 0.45, 0.55]) # 4 junctions
maxDif.append([0.50, 0.60, 0.45, 0.55])
minDif.append([0.30, 0.40, 0.40, 0.45]) # 5 junctions
maxDif.append([0.50, 0.50, 0.40, 0.40, 0.50])
minDif.append([0.28, 0.28, 0.32, 0.32, 0.42]) # 6 junctions

#Varshni, Y. P. Temperature dependence of the energy gap in semiconductors. Physica 34, 149 (1967)
def Varshni(T): #Gives gap correction in eV relative to 300K using GaAs parameters. T in K
    return (T**2)/(T+572)*-8.871e-4+0.091558486 # Beware, using GaAs parameters slightly overestimates the effect for most other semiconductors

def load(fname): 
    """ Load previously saved effs object. tandems.load('/path/and file name here') """
    with open(fname, "r") as f:
        t0 = f.readlines()
        t0 = ''.join(t0)
        return json_tricks.loads(t0)

class effs(object):
    """ Object class to hold results sets of yearly average photovoltaic efficiency 
        
    ---- USAGE EXAMPLE ----

    import tandems

    tandems.docs()    # Shows help file

    eff = tandems.effs(junctions=4, bins=6,  concentration=500)    #    Include as many or as few options as needed.
    eff.findGaps()
    eff.plot() # Figures saved to PNG files.
    
    eff.save() # Data saved for later reuse/replotting. Path and file name set in eff.name, some parameters and timestamp are appended to filename

    eff2 = tandems.copy.deepcopy(eff)
    eff2.__init__(bins=8, concentration=1, R=4e-5)  # Change some input parameters but keep previously found set of optimal gap combinations.
    eff2.recalculate() # Recalculate efficiencies for previously found set of optimal gap combinations.
    eff2.compare(eff) # Compares efficiencies in two datasets by doing eff2 - eff. Plots difference and saves PNG files.
    
    eff = tandems.load('/path/and file name here') # Load previusly saved data
    eff.results()
    eff.plot()
    """

    # s = self = current object instance
    
    # ---- Input variables ----
    
    junctions = 6
    topJunctions = 0 # Number of series conected juctions in top stack (topJunctions = 0 in 2 terminal devices)
    concentration = 1000
    gaps = [0, 0, 0, 0, 0, 0] # If a gap is 0, it is randomly chosen by tandems.findGaps(), otherwise it is kept fixed at value given here.
    ERE = 0.01 #external radiative efficiency without mirror. With mirror ERE increases by a factor (1 + beta)
    beta = 11 #n^2 squared refractive index  =  radiative coupling parameter  =  substrate loss.
    bins = 8 # bins is number of spectra used to evaluate eff, an array can be used to test the effect of the number of spectral bins. See convergence = True. 
    Tmin = 15+273.15 # Minimum ambient temperature at night in K
    deltaT = np.array([30, 55]) # Device T increase over Tmin caused by high irradiance (1000 W/m2), first value is for flat plate cell, second for high concentration cell
    convergence = False # Set to True to test the effect of changing the number of spectral bins  used to calculate the yearly average efficiency
    transmission = 0.02 # Subcell thickness cannot be infinite, 3 micron GaAs has transmission in the 2 to 3 % range (depending on integration range)
    thinning = False # Automatic subcell thinning for current matching
    thinSpec = 1 # Spectrum used to calculate subcell thinning for current matching. Integer index in specs array. 
    effMin = 0.02 # Lowest sampled efficiency value relative to maximum efficiency. Gaps with lower efficiency are discarded.
    d = 1 # 0 for global spectra, 1 for direct spectra
    # T = 70 for a 1mm2 cell at 1000 suns bonded to copper substrate. Cite I. Garcia, in CPV Handbook, ed. by: I. Rey-Stolle, C. Algora
    name = './Test' # Can optionally include path to destination of generated files. Example: "/home/documents/test". Some parameters and timestamp are appended to filename
    cells = 1000 # Desired number of calculated tandem cells. Will not exactly match number of returned results.
    R = 5e-7 # Series resistance of each stack in Ohm*m2. Default is optimistic value for high concentration devices
    # R = 4e-5 is suggested as optimistic value for one sun flat plate devices
    EQE = 0.7453*np.exp(-((Energies-1.782)/1.384)**4)+0.1992 # EQE model fitted to current record device, DOI.: 10.1109/JPHOTOV.2015.2501729
    mirrorLoss = 1 # Default value = 1 implies the assumption that back mirror loss = loss due to an air gap.
    opticallyCoupledStacks = False # Bottom junction of the top terminal stack can either have photon recycling or radiative coupling to the botttom stack. 
    coe = 0.9 # Concentrator optical efficiency. Optimistic default value. Used only for yield calculation.
    cloudCover = 0.26 # Fraction of the yearly energy that is lost due to clouds. Location dependent, used only for yield calculation. Default value 0.26 is representative of area near Denver, CO.
    # If using experimental spectra, set cloudCover = 0. If temporal resolution is low, it might be appropriate to set Tmin = Tmin + deltaT to keep T constant.
    specsFile = 'lat40.npy' # Name of the file with the spectral set obtained from tandems.generate_spectral_bins(). See genBins.py
    
    # ---- Results ----
    
    rgaps = 0 # Results Array with high efficiency Gap combinations found by trial and error
    Is = 0 # Results Array with Currents as a function of the number of spectral bins, 0 is standard spectrum 
    effs = 0 # Results Array with Efficiencies as a function of the number of spectral bins, 0 is standard spectrum
    
    # ---- Internal variables ----
    
    Irc = 0 # Radiative coupling current
    Itotal = 0 # Isc
    Pout = 0 # Power out
    Ijx = 0 # Array with the external photocurrents integrated from spectrum. Is set by getIjx()
    T = 0 # Set from irradiance at run time
    auxEffs = 0 # Aux array for efficiencies. Has the same shape as rgaps for plotting and array masking. 
    auxIs = 0 # Aux array for plotting. sum of short circuit currents from all terminals.
    specs = [] # Spectral set loaded from file
    P = [] # Array with integrated power in each spectrum
    Iscs = [] # Array with current in each spectrum
    thinTrans = 1 # Array with transmission of each subcell
    timeStamp = 0
    daytimeFraction = 1
    
    # ---- Methods and functions ----
    
    def __init__(s, **kwargs):
        """ Call this to change input parameters without discarding previously found gaps before calling recalculate() """ 
        s.timeStamp = time.time()
        for k, v in kwargs.items():
            setattr(s, k, v)
        if type(s.bins)==int:
            s.bins = [s.bins]
            
        s.specs = np.load(s.specsFile) # Load binned spectra
        s.daytimeFraction = s.specs[0, 0, -1] # This number is needed to calculate the yearly averaged power yield including night time hours. 
        s.specs[0, 0, :] = g1_5
        s.specs[1, 0, :] = d1_5
        s.Iscs = np.copy(s.specs)
        s.P = np.zeros((2, 46))
            
        def integra(d, spec): # Function to Integrate spectra from UV to given wavelength 
            s.P[d, spec] = integrate.trapz(s.specs[d, spec, :], x=wavel) # Power per unit area ( W / m2 )
            s.Iscs[d, spec, :] = (q/hc)*np.insert(1e-9*integrate.cumtrapz(s.EQE*s.specs[d, spec, :]*wavel, x=wavel), 0, 0) # Current per unit area ( A / m2 ), wavelength in nm
        for d in [0, 1]: # Integrate spectra from UV to given wavelength 
            for i in range(0, 46):
                integra(d, i)
        if s.topJunctions==0:
            s.topJunctions = s.junctions
        if s.convergence:
            s.bins = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        if s.concentration>1:
            s.d = 1 # use direct spectra
        else:
            s.d = 0 # use global spectra
            s.coe = 1 # concentrator optical efficiency is irrelevant in this case
            
    def intSpec(s, energy, spec): 
        """ Returns integrated photocurrent from given photon energy to UV """
        return np.interp(1e9*hc/energy/q, wavel, s.Iscs[s.d, spec, :]) # interpolate integrated spectra. Wavelength in nm
    def getIjx(s, spec):
        """ Get current absorbed in each junction, external photocurrent including junction transmission due to finite thickness """
        s.Ijx = np.zeros(s.junctions)
        IfromTop = 0
        upperIntSpec = 0
        for i in range(s.junctions-1, -1, -1): # From top to bottom: get external photocurrent in each junction
            IntSpec = s.intSpec(s.gaps[i]+Varshni(s.T), spec) # IntSpec = Integrated current from UV to given Energy gap
            Ijx0 = s.concentration*(IntSpec-upperIntSpec) # Get external I per junction 
            upperIntSpec = IntSpec
            if i != 0:
                s.Ijx[i] = (1-s.transmission)*Ijx0+IfromTop # Subcell thickness cannot be infinite, 3 micron GaAs has transmission in the 2 to 3 % range (depending on integration range)
                IfromTop = s.transmission*Ijx0
            else:
                s.Ijx[i] = Ijx0+IfromTop # bottom junction does not transmit (back mirror)
    def thin(s, topJ, bottomJ): 
        """ Calculate transmission factors for each subcell in series to maximize current under spectrum given in .thinSpec """
        # Top cell thinning: 
        # - From top to bottom junction
        #       - If next current is lower:
        #             - Find average with next junction
        #             - If average I is larger than next junction I, extend average and repeat this step
                
        s.getIjx(s.thinSpec) # get external photocurrent  
        initialIjx = np.copy(s.Ijx)
        
        Ijxmin = 0
        while int(Ijxmin*100) != int(s.Ijx.min()*100): # While min I keeps going up
            Ijxmin = s.Ijx.min()
            i = topJ
            while i > bottomJ-1: # Spread I from top to bottom
                if s.Ijx[i] > s.Ijx[i-1]: # If next current is lower 
                    imean = i
                    mean = s.Ijx[i]
                    previousMean = 0
                    while mean > previousMean:
                        imean -= 1
                        previousMean = mean
                        if imean > -1:
                            mean = np.mean(s.Ijx[imean:i+1])
                    s.Ijx[imean:i+1] = mean
                    i = imean
                i -= 1
        s.thinTrans = s.Ijx/initialIjx

        #print ('in', initialIjx, initialIjx.sum())
        #print ('out', s.Ijx, s.Ijx.sum())
        #print (s.thinTrans)
        
    def serie(s, topJ, bottomJ): 
        """ Get power from series connected subcells with indexes topJ to bottomJ. topJ = bottomJ is single junction. """
        # 1 - Get external photocurrent in each junction
        # 2 - Get current at the maximum power point from min external photocurrent
        # 3 - Add radiative coupling, recalculate maximum power point, this changes radiative coupling, repeat until self consistency
        # 4 - Calculate power out
        
        # Do "tandems.show_assumptions()" to see EQE model used and some characteristics of the spectral set used.

        if (topJ<0):
            return
        kT = con.k*s.T
        Irc0 = s.Irc #radiative coupling from upper stack 

        Ijx = s.Ijx*s.thinTrans # thinTrans is set by thin()
        Imax = Ijx[bottomJ:topJ+1].min() # Initial guess for current at the maximum power point 
        Imaxs = [0, Imax]
        while (((Imax-Imaxs[-2])/Imax)**2)>1e-7: # Loop to s consistently refine max power point
            V = 0
            s.Irc = Irc0 # Radiative coupling from upper stack 
            Ij = np.copy(Ijx) # Current absorbed in each junction
            for i in range(topJ, bottomJ-1, -1): # From top to bottom: get photocurrent in each junction including radiative coupling
                Ij[i] += s.Irc # Include emitted light from upper junction
                if (Ij[i]>Imax): # If there is excess current in this junction, radiative coupling
                    s.Irc = s.beta*s.ERE*(Ij[i]-Imax) #radiative coupling 
                else:
                    s.Irc = 0
            Ijmin = Ij[bottomJ:topJ+1].min() # Min current in series connected stack
            I = Ijmin*np.arange(0.8, 1, 0.0001) # IV curve sampling   
            for i in range(topJ, bottomJ-1, -1): # From top to bottom: Sample IV curve, get I0
                if (i == bottomJ) and not s.opticallyCoupledStacks: # The bottom junction of each series connected stack has photon recycling due to partial back reflection of luminescence
                    backLoss = s.mirrorLoss # This is the electroluminescence photon flux lost at the back of the bottom junction normalized to ERE
                else:
                    backLoss = s.beta # This is the electroluminescence photon flux lost to the next lower junction
                I0 = (1+backLoss) * 2*np.pi*q * np.exp(-1*s.gaps[i]*q/kT) * kT**3*((s.gaps[i]*q/kT+1)**2+1) / (con.h*hc**2) / s.ERE # Dark current at V = 0 in A / m2 s
                V += (kT/q)*np.log((Ij[i]-I)/I0+1) # add voltage of series connected cells
            V -= s.R*I # V drop due to series resistance
            Imax = I[np.argmax(I*V)] # I at max power point
            Imaxs.append(Imax)
        if len(Imaxs)>10:
            print ('s consistency is slowing convergence while finding the maximum power point.')
            print ('ERE or beta might be too high.')
            print ('Current at the maximum power point is converging as:', Imaxs)
            pdb.set_trace()
        s.Itotal += Ijmin
        s.Pout += (I*V).max()
        
    def stack(s, spec): 
        """ Use a single spectrum to get power from 4 terminal tandem. If topJunctions = junctions the result is for 2 terminal tandem. """
        s.Irc = 0 # For top cell there is no radiative coupling from upper cell
        s.T = s.Tmin+s.deltaT[s.d]*s.P[s.d, spec]/1000 # To a first approximation, cell T is a linear function of irradiance.
        # T = 70 for a 1mm2 cell at 1000 suns bonded to copper substrate. Cite I. Garcia, in CPV Handbook, ed. by: I. Rey-Stolle, C. Algora
        s.getIjx(spec) # Get external photocurrents
        s.serie(s.junctions-1, s.junctions-s.topJunctions) # Add efficiency from top stack, topJunctions is number of juntions in top stack
        if not s.opticallyCoupledStacks:
            s.Irc = 0
        s.serie(s.junctions-s.topJunctions-1, 0) # Add efficiency from bottom stack
        return
    def findGaps(s): 
        """ Calculate efficiencies for random band gap combinations. """
        startTime = time.time()
        ncells = 0 # Number of calculated gap combinations
        nres = 0
        effmax = 0
        Emin_ = np.array(Emin[s.junctions-1]) # Initial guess at where the eff maxima are. 
        Emax_ = np.array(Emax[s.junctions-1]) # The search space is expanded as needed if high eff are found at the edges of the search space.        
        minDif_ = np.array(minDif[s.junctions-2])
        maxDif_ = np.array(maxDif[s.junctions-2])
        # REMOVE SEED unless you want to reuse the same sequence of random numbers (e.g.: compare results after changing one parameter)
        #np.random.seed(07022015)
        s.rgaps = np.zeros((s.cells+1000, s.junctions)) # Gaps
        s.auxIs = np.zeros((s.cells+1000, s.junctions)) # Aux array for plotting only
        s.auxEffs = np.zeros((s.cells+1000, s.junctions)) # Aux array for plotting only  
        s.Is = np.zeros((s.cells+1000, 10)) # Currents as a function of the number of spectral bins, 0 is standard spectrum 
        s.effs = np.zeros((s.cells+1000, 10)) # Efficiencies as a function of the number of spectral bins, 0 is standard spectrum 
        fixedGaps = np.copy(s.gaps) # Copy input gaps to remember which ones are fixed, if gap==0 make it random
        while (nres<s.cells+1000): # Loop to randomly sample a large number of gap combinations
            s.gaps = np.zeros(s.junctions)  
            lastgap = 0
            i = 0
            while i<s.junctions:     # From bottom to top: define random gaps
                if i>0:
                    Emini = max(Emin_[i], lastgap + minDif_[i-1]) # Avoid gap combinations that are too unevenly spaced
                    Emaxi = min(Emax_[i], lastgap + maxDif_[i-1])
                else:
                    Emini = Emin_[i]
                    Emaxi = Emax_[i]
                Erange = Emaxi - Emini
                if fixedGaps[i] == 0:
                    s.gaps[i] = Emini + Erange * np.random.rand() #define random gaps
                else:
                    s.gaps[i] = fixedGaps[i] # Any gaps with a value other than 0 are kept fixed, example: tandems.effs(gaps=[0.7, 0, 0, 1.424, 0, 2.1]) .
                if i>0 and fixedGaps.sum(): # If there are fixed gaps check gaps are not too unevenly spaced
                    if not ( 0.1 < s.gaps[i]-lastgap < 1 ): # gap difference is not in range
                        lastgap = 0 # Discard gaps and restart if gaps are too unevenly spaced
                        i = 0
                    else:
                        lastgap = s.gaps[i]
                        i += 1
                else:
                    lastgap = s.gaps[i]
                    i += 1
            
            if s.thinning:
                bottomJts = s.junctions-s.topJunctions # index number for the bottom junction of the top stack
                s.thin(s.junctions-1, bottomJts) # Find optimal subcell thinning for top stack for a certain spectrum
                if bottomJts>0: # if there is a lower stack
                    exIb = s.Ijx[bottomJts]-s.Ijx[bottomJts:].min() # The excess current from the bottom junction of the top stack, goes to lower stack
                    s.Ijx[bottomJts] -= exIb
                    s.Ijx[bottomJts-1] += exIb
                    s.thin(bottomJts-1, 0) # Find optimal subcell thinning for bottom stack. 
                
            s.Itotal = 0
            s.Pout = 0
            s.stack(1) # calculate power out with average spectrum (1 bin).
            eff = s.Pout/s.P[s.d, 1]/s.concentration # get efficiency
            
            if (eff>effmax-s.effMin-0.01): # If gap combination is good, do more work on it
                if nres % 10 == 0: # Show calculation progress from time to time
                    print ('Tried '+str(ncells)+', got '+str(nres)+' candidate gap combinations.', end="\r")
                for gi, gap in enumerate(s.gaps): # Expand edges of search space if any of the found gaps are near an edge
                    if gap < Emin_[gi] + 0.01: # Expand energy range allowed for each gap as needed
                        if Emin_[gi] > 0.5: # Only energies between 0.5 and 2.5 eV are included
                            Emin_[gi] -= 0.01
                    if gap > Emax_[gi] - 0.01:
                        if Emax_[gi] < 2.5: 
                            Emax_[gi] += 0.01
                    if gi > 0:
                        if gap - s.gaps[gi-1] + 0.01 > maxDif_[gi-1]: # Allow for more widely spaced gaps if needed
                            maxDif_[gi-1] += 0.01
                        if gap - s.gaps[gi-1] - 0.01 < minDif_[gi-1]: # Allow for more closely spaced gaps if needed
                            minDif_[gi-1] -= 0.01
                if (eff>effmax):
                    effmax = eff
                s.rgaps[nres, :] = s.gaps
                s.effs[nres, 1] = eff 
                s.Is[nres, 1] = s.Itotal 
                for i in s.bins: # loop for number of bins
                    s.Itotal = 0
                    Pin = 0
                    s.Pout = 0                  
                    for j in range(0, a1123456789[i]): # loop for bin index.    a1123456789 = [1, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                        s.stack(bindex[i][j]) # calculate power out for each spectra and add to Pout to calculate average efficiency
                        Pin += s.P[s.d, bindex[i][j]]
                    s.effs[nres, i] = s.Pout/Pin/s.concentration
                    s.auxEffs[nres, :] = np.zeros(s.junctions)+s.effs[nres, i]
                    s.auxIs[nres, :] = np.zeros(s.junctions)+s.Itotal/a1123456789[i] # a1123456789 = [1, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                    s.Is[nres, i] = s.Itotal/a1123456789[i]
                nres += 1
                                
            ncells += 1
        
        mask = s.auxEffs > s.auxEffs.max()-s.effMin
        s.rgaps = s.rgaps[mask] # Discard results below the efficiency threshold set by effMin
        s.auxIs = s.auxIs[mask] # As a side effect of this cut off, arrays are flattened
        s.auxEffs = s.auxEffs[mask] # [:, 0]
        threshold = s.effs[:, s.bins[-1]].max()-s.effMin
        mask = s.effs[:, s.bins[-1]] > threshold
        s.effs = s.effs[mask]
        s.Is = s.Is[mask]
        
        s.rgaps = np.reshape(s.rgaps, (-1, s.junctions)) # Restore proper array shapes after masking 
        s.auxIs = np.reshape(s.auxIs, (-1, s.junctions))
        s.auxEffs = np.reshape(s.auxEffs, (-1, s.junctions))
        s.effs = np.reshape(s.effs, (-1, 10))
        s.Is = np.reshape(s.Is, (-1, 10))
        
        tiempo = int(time.time()-startTime)
        res = np.size(s.rgaps)/s.junctions
        print ('Calculated ', ncells, ' and saved ', res, ' gap combinations in ', tiempo, ' s :', res/(tiempo+1), ' results/s')
        s.results()
    def kWh(s, efficiency):
        """Converts efficiency values to yearly energy yield in kWh/m2"""
        return 365.25*24 * s.P[s.d,1] * efficiency * (1 - s.cloudCover) * s.daytimeFraction / 1000
    def results(s):
        """ After findGaps() or recalculate(), or load(), this function shows the main results """
        print ('Maximum efficiency:', s.auxEffs.max())
        print ('Maximum yearly energy yield:', s.kWh(s.auxEffs.max()))
        imax = np.argmax(s.auxEffs[:, 0])
        print ('Optimal gaps:', s.rgaps[imax])
        print ('Isc for optimal gaps (A/m2):', s.auxIs[imax, 0])
        print ('Isc range (A/m2):', s.auxIs.min(), '-', s.auxIs.max())
    def plot(s):
        """ Saves efficiency plots to PNG files """ 
        res = np.size(s.rgaps)/s.junctions
        Is_ = np.copy(s.auxIs)
        Is_ = (Is_-Is_.min())/(Is_.max()-Is_.min())
        srIs_ = (s.Is[:, s.bins[-1]]-s.Is[:, s.bins[-1]].min())/(s.Is[:, s.bins[-1]].max()-s.Is[:, s.bins[-1]].min())
        if s.convergence:
            plt.figure()
            plt.xlabel('Number of spectral bins')
            plt.ylabel('Absolute efficiency change \n by increasing number of bins (%)')
            for i in range(0, int(res)):
                diffs = []
                for j in s.bins[:-1]:
                    diffs.append(100*(s.effs[i, j+1]-s.effs[i, j]))
                plt.plot(s.bins[:-1], diffs, color = LGBT(srIs_[i]), linewidth=0.1) #  color = LGBT(auxIs[i*s.junctions])
            plt.savefig(s.name+' '+s.specsFile.replace('.npy', '')+' convergence '+str(int(s.junctions))+' '+str(int(s.topJunctions))+' '+str(int(s.concentration))+' '+str(int(s.timeStamp)), dpi=600)
            plt.show()
        plt.figure()
        ymin = (s.effs[:, s.bins[-1]].max()-s.effMin)
        ymax = s.effs[:, s.bins[-1]].max() + 0.001
        plt.ylim(100*ymin, 100*ymax)
        plt.xlim(0.5, 2.5)
        #plt.grid(True)
        plt.minorticks_on()
        plt.tick_params(direction='out', which='minor')
        plt.tick_params(direction='inout', pad=6)
        plt.ylabel('Yearly averaged efficiency (%)')
        plt.xlabel('Energy gaps (eV)')
        plt.plot([0.50, 0.50], [0, 100], c='grey', linewidth=0.5)
        plt.text(0.5, 100*s.auxEffs.max()+.2, 'A')
        plt.plot([0.69, 0.69], [0, 100], c='grey', linewidth=0.5)
        plt.text(0.69, 100*s.auxEffs.max()+.2, 'B')
        plt.plot([0.92, 0.92], [0, 100], c='grey', linewidth=0.5)
        plt.text(0.92, 100*s.auxEffs.max()+.2, 'C')
        plt.plot([1.1, 1.1], [0, 100], c='grey', linewidth=0.5)
        plt.text(1.1, 100*s.auxEffs.max()+.2, 'D')
        plt.plot([1.33, 1.33], [0, 100], c='grey', linewidth=0.5)
        plt.text(1.33, 100*s.auxEffs.max()+.2, 'E1')
        plt.plot([1.63, 1.63], [0, 100], c='grey', linewidth=0.5)
        plt.text(1.63, 100*s.auxEffs.max()+.2, 'E2')
        plt.scatter(s.rgaps, 100*s.auxEffs, c=Is_, s=70000/len(s.effs[:, s.bins[-1]]), edgecolor='none', cmap=LGBT)
        axb = plt.gca().twinx()
        axb.set_ylim(s.kWh(ymin),s.kWh(ymax))
        axb.set_ylabel('Yield $\mathregular{(kWh / m^2 / year)}$')
        plt.savefig(s.name+' '+s.specsFile.replace('.npy', '')+' '+str(int(s.junctions))+' '+str(int(s.topJunctions))+' '+str(int(s.concentration))+' '+str(int(s.timeStamp)), dpi=600)
        plt.show()        
        plt.xlim(100*(s.auxEffs.max()-s.effMin), 100*s.auxEffs.max())
        plt.xlabel('Yearly averaged efficiency (%)')
        plt.ylabel('Count')
        plt.hist(100*s.effs[:, s.bins[-1]], bins=30)
        plt.savefig(s.name+' Hist '+s.specsFile.replace('.npy', '')+' '+str(int(s.junctions))+' '+str(int(s.topJunctions))+' '+str(int(s.concentration))+' '+str(int(s.timeStamp)), dpi=600)      
        plt.show()
    def save(s):
        """ Saves data for later reuse/replotting. Path and file name set in eff.name, some parameters and timestamp are appended to filename """
        with open(s.name+' '+s.specsFile.replace('.npy', '')+' '+str(int(s.junctions))+' '+str(int(s.topJunctions))+' '+str(int(s.concentration))+' '+str(int(s.timeStamp)), "w") as f:
            f.write(json_tricks.dumps(s))
    def recalculate(s):
        """ Recalculate efficiencies with new set of input parameters using previously found gaps. Call __init__() to change parameters before recalculate() """
        for gi in range(0, s.rgaps.shape[0]):
            s.gaps = s.rgaps[gi]
            for i in s.bins: # loop for number of bins
                s.Itotal = 0
                Pin = 0
                s.Pout = 0                  
                for j in range(0, a1123456789[i]): # loop for bin index.    a1123456789 = [1, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                    s.stack(bindex[i][j]) # calculate power out for each spectra and add to Pout to calculate average efficiency
                    Pin += s.P[s.d, bindex[i][j]]
                s.effs[gi, i] = s.Pout/Pin/s.concentration
                s.auxEffs[gi, :] = np.zeros(s.junctions)+s.effs[gi, i]
                s.auxIs[gi, :] = np.zeros(s.junctions)+s.Itotal/a1123456789[i] # a1123456789 = [1, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                s.Is[gi, i] = s.Itotal/a1123456789[i]
        s.results()
    def compare(s, s0): 
        """ Plots relative differences of effiency for two results sets based on the same set of optimal band gap combinations. """
        print ('I min, I max : ', s.auxIs.min(), s.auxIs.max())
        print ('eff min, eff max : ', s.auxEffs.min(), s.auxEffs.max())
        Is_ = np.copy(s0.auxIs)
        Is_ = (Is_-Is_.min())/(Is_.max()-Is_.min())

        plt.figure()
        plt.minorticks_on()
        plt.tick_params(direction='out', which='minor')
        plt.tick_params(direction='inout', pad=6)
        if s.specsFile == s0.specsFile:
            diff = 100*(s.auxEffs - s0.auxEffs)
            plt.ylabel('Efficiency change (%)')
        else: # It does not make much sense to compare efficiencies if spectra are different, compare energy yield instead
            diff = s.kWh(s.auxEffs) - s0.kWh(s0.auxEffs)
            plt.ylabel('Yield change $\mathregular{(kWh / m^2 / year)}$')
        plt.xlabel('Current $\mathregular{(A/m^2)}$')
        plt.scatter(s0.auxIs , diff , c=Is_ , s=20000/len(s.effs[:, s.bins[-1]]) , edgecolor='none', cmap=LGBT)
        plt.savefig(s.name+'-'+s0.name.replace('/', '').replace('.', '')+' '+s.specsFile.replace('.npy', '').replace('/', '').replace('.', '')+'-'+s0.specsFile.replace('.npy', '').replace('/', '').replace('/', '')+' '+str(int(s0.junctions))+' '+str(int(s0.topJunctions))+' '+str(int(s0.concentration))+' '+str(int(s.timeStamp)), dpi=600)
        plt.show()        

# ---- End of Class effs ----

def show_assumptions(): # Shows the used EQE model and the AOD and PW statistical distributions 
    s0 = effs()
    plt.figure()
    plt.xlim(0.4, 4)
    plt.ylim(0, 1)
    plt.title('EQE model fitted to record device with 46% eff. \n DOI.: 10.1109/JPHOTOV.2015.2501729')
    plt.ylabel('External Quantum Efficiency')
    plt.xlabel('Photon energy (eV)')
    plt.plot(Energies, s0.EQE)
    plt.show()
    
    #AOD random distribution used here is fitted to histograms by Jaus and Gueymard based on Aeronet data
    lrnd = np.random.lognormal(-2.141, 1, size=1000000)
    lrnd[lrnd>2] = 2 # For plotting purposes, AOD values are capped here at AOD=2, this is not done during calculations
    plt.figure()
    titel = 'AOD random distribution used here is fitted to \n histograms by Jaus and Gueymard based on Aeronet data, \n'
    titel += 'DOI: 10.1063/1.4753849 \n In Jaus and Gueymard, percentiles at 25% and 75% are 0.06, 0.23. \n'
    titel += 'Here: '+str(np.percentile(lrnd, 25))+' '+str(np.percentile(lrnd, 75))
    plt.title(titel)
    plt.xlim(0, 0.8)
    plt.xlabel('Aerosol Optical Depth at 500 nm')
    plt.hist(lrnd, bins=200)
    plt.show()
    
    #PW random distribution used here is fitted to histograms by Jaus and Gueymard based on Aeronet data
    lrnd = 0.39*np.random.chisquare(4.36, size=1000000)
    lrnd[lrnd>7] = 7 # For plotting purposes, PW values are capped here at AOD=2, this is not done during calculations
    plt.figure()
    titel = 'PW random distribution used here is fitted to \n histograms by Jaus and Gueymard based on Aeronet data, \n'
    titel += 'DOI: 10.1063/1.4753849 \n In Jaus and Gueymard, percentiles at 25% and 75% are 0.85, 2.27. \n'
    titel += 'Here: '+str(np.percentile(lrnd, 25))+' '+str(np.percentile(lrnd, 75))
    plt.title(titel)
    plt.xlim(0, 6)
    plt.xlabel('Precipitable water (cm)')
    plt.hist(lrnd, bins=40)
    plt.show()

def generate_spectral_bins(latMin=40, latMax=40, longitude='random', 
        AOD='random', PW='random', tracking='38 -999 -999', 
        speCount = 20000, NSRDBfile='', fname='Iscs'):
    def EPR(i, s1): 
        """ Calculates EPR, integrates Power and stores spectra"""
        P[:,i] = integrate.trapz(s1, x=wavel, axis=1)
        if (P[0, i] > 0) and (P[1, i] > 0):
            specs[:, i, :] = s1
            EPR650[:, i] = integrate.trapz( s1[:, :490], x=wavel[:490], axis=1 )
            EPR650[:, i] = ( P[:, i] - EPR650[:, i] ) / EPR650[:, i] # Calculate EPR, Ivan Garcia's criteria for binning
            i += 1
        return i
            
    # NECESARY CHANGES IN SMARTS 2.9.5 SOURCE CODE
    # Line 189
    #       batch = .TRUE.
    # Line 1514
    #      IF(Zenit.LE.75.5224)GOTO 13
    #      WRITE(16, 103, iostat = Ierr24)Zenit
    # 103  FORMAT(//,'Zenit  =  ',F6.2,' is > 75.5224 deg. (90 in original code) This is equivalent to AM < 4'
    # This change is needed because trackers are shadowed by neighboring trackers when the sun is near the horizon. 
    # Zenit 80 is already too close to the horizon to use in most cases due to shadowing issues.
    # tracking='38 -999 -999' for 2 axis tracking. tracking='38 37 -999' for 1 axis tracking. '38 37 180' for fixed panel. 
    # 37 is near optimal tilt only for a certain range of lattitudes. 180 is for northern hemisphere.

    specs = np.zeros((2, speCount, len(wavel))) # Array to hold generated spectra
    EPR650 = np.zeros((2, speCount)) # Ivan Garcia's criteria for binning, store EPR value for each spectrum
    P = np.zeros((2, speCount)) # Power for each spectrum

    if NSRDBfile != '': # Optionally Load specified National Solar Resource DataBase spectra
        # https://maps.nrel.gov 
        Ng = np.loadtxt(NSRDBfile, skiprows=3, delimiter=',', usecols=tuple(range(159, 310))) # Global Horizontal
        Nd = np.loadtxt(NSRDBfile, skiprows=3, delimiter=',', usecols=tuple(range(8, 159))) # Direct spectra
        attempts = Nd.shape[0]
        if fname == 'Iscs':
            fname = NSRDBfile.replace('.csv', '')
        s1 = np.zeros((2, len(wavel)))
        speCount = 0
        for i in range(0, len(Nd)):
            s1[0,:] = np.interp(wavel, np.arange(300, 1810, 10), Ng[i, :], left=0, right=0) # Global Horizontal
            s1[1,:] = np.interp(wavel, np.arange(300, 1810, 10), Nd[i, :], left=0, right=0) # Direct spectra
            speCount = EPR(speCount, s1) # Calculates EPR, integrates Power and stores spectra
            print (speCount, 'spectra out of', attempts, 'points in time', end="\r")
        specs = specs[:,:speCount] # Array to hold generated spectra
        EPR650 = EPR650[:,:speCount] # Ivan Garcia's criteria for binning, store EPR value for each spectrum
        P = P[:,:speCount] # Power for each spectrum        
    else:
        longitude2 = longitude
        AOD2 = AOD
        PW2 = PW
        #os.chdir('/path to your SMARTS files')
        with open ('smarts295.in_.txt', "r") as fil: # Load template for SMARTS input file
            t0 = fil.readlines()
        t0 = ''.join(t0)
        attempts = 0
        t0 = t0.replace('38 -999 -999', tracking)
        for i in range(0, speCount):
            tama = 0
            while tama == 0: # Generate random time, location, AOD, PW
                
                if longitude == 'random':
                    longitude2 = 360*np.random.rand()-180
                if AOD == 'random':
                    AOD2 = min(5.4, np.random.lognormal(-2.141, 1)) # For more info do "tandems.show_assumptions()"
                if PW == 'random':
                    PW2 = min(12, 0.39*np.random.chisquare(4.36))
                
                t1 = t0.replace('AOD#', str(AOD2)) # Substitute AOD and PW into template SMARTS input file
                t1 = t1.replace('PW#', str(PW2)) 
                t2 = datetime.fromtimestamp(3155673600.0*np.random.rand()+1517958000.0).strftime("%Y %m %d %H")+' ' # from 7/feb/2018 to 7/feb/2018
                t2 += '%.2f' % ((latMax-latMin)*np.arccos(2*np.random.rand()-1)/np.pi+latMin)+' '  # 50 > Latitude > -50
                t2 += '%.2f' % (longitude2)+' 0\t\t\t!Card 17a Y M D H Lat Lon Zone\n\n'
                with open('smarts295.inp.txt' , "w") as fil:
                    fil.write(t1+t2)
                try:
                    os.remove("smarts295.ext.txt")
                except OSError:
                    pass
                try:
                    os.remove("smarts295.out.txt")
                except OSError:
                    pass                
                attempts += 1
                ou = sub.check_call('./smartsAM4', shell=True) # execute SMARTS
                tama = os.stat('smarts295.ext.txt').st_size # check output file
                if tama>0:
                    try: # load output file if it exists
                        s1 = np.loadtxt("smarts295.ext.txt", delimiter=' ', usecols=(0, 1, 2), unpack=True, skiprows=1)#Global Tilted = 1, Direct Normal= 2, Diffuse Horizontal=3
                    except:
                        tama = 0
            EPR(i, s1[1:,:]) # Calculates EPR, integrates Power and stores spectra
        print ('Finished. Called smarts '+str(attempts)+' times and got '+str(speCount)+' spectra')

    totalPower = np.zeros(2)
    bincount = np.zeros((2,9), dtype=np.int)
    binlimits = []
    binspecs = []
    Iscs = np.zeros((2, 46, len(wavel))) # Prepare data structure for saving to file
    for m in (0,1):
        specs[m, :] = specs[m, np.argsort(EPR650[m,:]), :] # Sort spectra and integrated powers by EPR
        P[m, :] = P[m, np.argsort(EPR650[m,:])]
        totalPower[m] = P[m, :].sum() # Sum power
        binlimits.append([])
        binspecs.append([])
        for i in range(0, 9): # Prepare data structure to store bin limits and averaged spectra
            binlimits[m].append(np.zeros(i+2, dtype=np.int))
            binspecs[m].append(np.zeros((i+1, len(wavel))))
        bincount[m] = np.zeros(9, dtype=np.int)
        accubin = 0 # Calculate bin limits with equal power in each bin
        for i in range(0, speCount):
            accubin += P[m, i] # Accumulate power and check if its a fraction of total power
            for j in range(0, 9):
                if accubin >= totalPower[m]/(j+1)*(bincount[m, j]+1): # check if its a fraction of total power
                    bincount[m, j] += 1
                    binlimits[m][j][bincount[m,j]] = i+1 # Store bin limit     
        for i in range(0, 9): #iterate over every bin set
            binlimits[m][i][-1] = speCount # set the last bin limit to the total number of spectra
        # Average spectra using the previously calculated bin limits
        for i in range(0, 9): #bin set
            for j in np.arange(0, i+1, 1): #bin counter
                binspecs[m][i][j,:] = np.sum(specs[m, binlimits[m][i][j]:binlimits[m][i][j+1]], axis=0) / (binlimits[m][i][j+1]-binlimits[m][i][j]) # Average spectra in each bin
        k = 1
        for i in range(0, 9): #bin set
            for j in range(0, i+1): #bin counter
                Iscs[m, k, :] = binspecs[m][i][j,:]
                k += 1
    
    Iscs[0, 0, -1] = speCount/attempts # This number is needed to calculate the yearly averaged power yield including night time hours. 
    
    np.save(fname, Iscs)
    
def docs(): # Shows HELP file
    with open('HELP', 'r') as fin:
        print (fin.read())
        
