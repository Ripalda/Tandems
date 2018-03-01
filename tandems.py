# Random sampling of multijunction photovoltaic efficiencies. Jose M. Ripalda
# Tested with Python 2.7 and 3.6
# Requires installing json_tricks
# SMARTS 2.9.5 and sklearn are required only to generate new sets of spectra.
# Provided files with ".npy" extension can be used instead to load a set of spectra.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time
import pdb
import json_tricks
from copy import deepcopy
import os.path
import subprocess as sub
from datetime import datetime
from scipy import integrate
from sklearn.cluster import FeatureAgglomeration
from sklearn.cluster import KMeans
from glob import glob
import scipy.constants as con
hc = con.h*con.c
q = con.e

version = 37
print ('Tandems version',version)

np.set_printoptions(precision=3) # Print 3 decimal places only

colors = [(1, 0, 1), (0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]  # B -> G -> R
LGBT = LinearSegmentedColormap.from_list('LGBT', colors, 500)

# Load standard reference spectra ASTM G173
wavel, g1_5, d1_5 = np.loadtxt("AM1_5 smarts295.ext.txt", delimiter=',', usecols=(0,1,2), unpack=True)
Energies = 1e9*hc/q/wavel # wavel is wavelenght in nm

specIndexArray = [] # "2D Array for spectral bin indexing. First index is number of bins, second is bin index. 
specIndexArray.append([0])
specIndex = 1
for numBins in range(1, 21): #bin set
    specIndexArray.append([])
    for binIndex in range(0, numBins): #bin counter
        specIndexArray[numBins].append(specIndex)
        specIndex += 1
        
def getSpectrumIndex(numBins,binIndex): # Returns a spectrum index for input in spectra array.
    if numBins>20: # Use full set of spectra
        return binIndex+211
    else: # Use averaged spectra
        return specIndexArray[numBins][binIndex]

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

def load(fnames): 
    """ Load previously saved effs objects. tandems.load('/path/and_file_name_pattern*here')
    A file name pattern with wildcards (*) can be used to load a number of files and join them in a single object. 
    This is useful for paralelization.
    All files need to share the same set of input parameters because this function
    does not check if it makes sense to join the output files."""
    s = False
    arrayOfFiles = glob(fnames)
    if arrayOfFiles:
        for fname in arrayOfFiles:
            with open(fname, "rb") as f:         
                if not s:
                    s = json_tricks.loads(f.read())
                else:
                    s2 = json_tricks.loads(f.read())
                    s.rgaps = np.append(s.rgaps,s2.rgaps, axis=0)
                    s.auxIs = np.append(s.auxIs,s2.auxIs, axis=0)
                    s.auxEffs = np.append(s.auxEffs,s2.auxEffs, axis=0)
                    s.Is = np.append(s.Is,s2.Is, axis=0)
                    s.effs = np.append(s.effs,s2.effs, axis=0)
        s.cells = s.rgaps.shape[0]
    else:
        print('Files not found')
    return s

def multiFind( cores = 2, saveAs = 'someParallelData', parameterStr = 'cells=1000, junctions=4'):
    """Hack to use as many CPU cores as desired in the search for optimal band gaps.
    Uses the tandems.effs object and its findGaps method.
    Total number of results will be cores * cells.
    Calls bash (so do not expect this to work under Windows) to create many instances of this python module running in parallel."""
    os.system('for i in `seq 1 '+str(cores)+'`; do python -c "import tandems;s=tandems.effs(name='+chr(39)+saveAs+chr(39)+','+parameterStr+');s.findGaps();s.save()" & done; wait')
    s = load(saveAs+'*') # Loads all results and joins them in a single object
    s.results()
    s.plot()

class effs(object):
    """ Object class to hold results sets of yearly average photovoltaic efficiency 
        
    ---- USAGE EXAMPLE ----

    import tandems

    tandems.docs()    # Shows help file

    eff = tandems.effs(junctions=4, bins=6,  concentration=500)    #    Include as many or as few options as needed.
    eff.findGaps()
    eff.plot() # Figures saved to PNG files.
    
    eff.save() # Data saved for later reuse/replotting. Path and file name set in eff.name, some parameters and autoSuffix are appended to filename

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
    bins = 15 # bins is number of spectra used to evaluate eff. See convergence = True. 
    Tmin = 15+273.15 # Minimum cell temperature at zero irradiance in K
    deltaT = np.array([30, 55]) # Device T increase over Tmin caused by high irradiance (1000 W/m2), first value is for one sun cell, second for high concentration cell
    convergence = False # Set to True to test the effect of changing the number of spectral bins used to calculate the yearly average efficiency
    transmission = 0.02 # Subcell thickness cannot be infinite, 3 micron GaAs has transmission in the 2 to 3 % range (depending on integration range)
    thinning = False # Automatic subcell thinning for current matching
    thinSpec = 1 # Spectrum used to calculate subcell thinning for current matching. Integer index in spectra array. 
    effMin = 0.02 # Lowest sampled efficiency value relative to maximum efficiency. Gaps with lower efficiency are discarded.
    d = 1 # 0 for global spectra, 1 for direct spectra
    # T = 70 for a 1mm2 cell at 1000 suns bonded to copper substrate. Cite I. Garcia, in CPV Handbook, ed. by: I. Rey-Stolle, C. Algora
    name = './Test' # Can optionally include path to destination of generated files. Example: "/home/documents/test". Some parameters and autoSuffix are appended to filename
    cells = 1000 # Target number of calculated tandem cells. Actual number of results will be less.
    R = 5e-7 # Series resistance of each stack in Ohm*m2. Default is optimistic value for high concentration devices
    # R = 4e-5 is suggested as optimistic value for one sun flat plate devices
    EQE = 0.7453*np.exp(-((Energies-1.782)/1.384)**4)+0.1992 # EQE model fitted to current record device, DOI.: 10.1109/JPHOTOV.2015.2501729
    mirrorLoss = 1 # Default value = 1 implies the assumption that back mirror loss = loss due to an air gap.
    opticallyCoupledStacks = False # Bottom junction of the top terminal stack can either have photon recycling or radiative coupling to the botttom stack. 
    coe = 0.9 # Concentrator optical efficiency. Optimistic default value. Used only for yield calculation.
    cloudCover = 0.26 # Fraction of the yearly energy that is lost due to clouds. Location dependent, used only for yield calculation. Default value 0.26 is representative of area near Denver, CO.
    # If using experimental spectra, set cloudCover = 0. If temporal resolution is low, it might be appropriate to set Tmin = Tmin + deltaT to keep T constant.
    specsFile = 'lat40.clusters.npy' # Name of the file with the spectral set obtained from tandems.generate_spectral_bins(). See genBins.py
    # File types for spectral sets should either be .clusters.npy or .bins.npy
    
    # ---- Results ----
    
    rgaps = np.zeros((1, 6)) # Results Array with high efficiency Gap combinations found by trial and error
    Is = np.zeros((1, 22)) # Results Array with Currents as a function of the number of spectral bins, 0 is standard spectrum 
    effs = np.zeros((1, 22)) # Results Array with Efficiencies as a function of the number of spectral bins, 0 is standard spectrum
    
    # ---- Internal variables ----
    
    Irc = 0 # Radiative coupling current
    Itotal = 0 # Isc
    Pout = 0 # Power out
    T = 300 # Set from irradiance at run time in Kelvin
    auxEffs = np.zeros((1, 6)) # Aux array for efficiencies. Has the same shape as rgaps for plotting and array masking. 
    auxIs = np.zeros((1, 6)) # Aux array for plotting. sum of short circuit currents from all terminals.
    spectra = [] # Spectral set loaded from file
    P = [] # Array with integrated power in each spectrum
    Iscs = [] # Array with current in each spectrum
    thinTrans = 1 # Array with transmission of each subcell
    autoSuffix = ''
    daytimeFraction = 1
    numSpectra = [1]+list(range(1,21))+[10000] # Number of spectra as a function of numBins
    timePerCluster = 0 # Fraction of the yearly daytime hours that is represented by each cluster of spectra. Set in __init__ 
    
    # ---- Methods and functions ----
    
    def __init__(s, **kwargs):
        """ Call __init__ to change input parameters without discarding previously found gaps before calling recalculate() """

        # This is for having an automatic unique suffix in filenames to reduce the chance of accidental overwriting
        alphabet='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        number = int((62**6-1)*np.random.rand())
        while number != 0:
            number, i = divmod(number, 62)
            s.autoSuffix = alphabet[i] + s.autoSuffix

        for k, v in kwargs.items():
            setattr(s, k, v)
        if type(s.bins)==int:
            s.bins = [s.bins]
        
        if '.clusters.npy' in s.specsFile:
            s.timePerCluster = np.load(s.specsFile.replace('.clusters.','.timePerCluster.')) # Fraction of the yearly daytime hours that is represented by each cluster of spectra
            s.timePerCluster[0, 0] = 1 # These are for the standard reference spectra ASTM G173.
            s.timePerCluster[1, 0] = 1
        elif '.bins.npy' not in s.specsFile:
            print('File types for spectral sets should either be .clusters.npy or .bins.npy')
            return            
        
        if s.cells == 1:
            s.rgaps = np.array([s.gaps])
            s.auxIs = np.zeros((s.cells, s.junctions)) # Aux array for plotting only
            s.auxEffs = np.zeros((s.cells, s.junctions)) # Aux array for plotting only  
            s.Is = np.zeros((s.cells, 22)) # Currents as a function of the number of spectral bins, 0 is standard spectrum, 10 is full spectral dataset
            s.effs = np.zeros((s.cells, 22)) # Efficiencies as a function of the number of spectral bins, 0 is standard spectrum, 10 is full spectral dataset
        
        s.spectra = np.load(s.specsFile) # Load binned or clustered spectra
        s.daytimeFraction = s.spectra[0, 0, -1] # This is the fraction of daytime hours in a year and is needed to calculate the yearly averaged power yield including night time hours. 
        s.spectra[0, 0, -1] = 0
        s.spectra[0, 0, :] = g1_5 # These are for the standard reference spectra ASTM G173.
        s.spectra[1, 0, :] = d1_5
        
        if s.convergence:
            s.bins =  list(range(1,20))+[21,20]  # bin 21 is the full set of spectra, bin 1 is the average
            fullSpecs = np.load(s.specsFile.replace('.clusters.npy','.full.npy').replace('.bins.npy','.full.npy'))
            s.spectra = np.concatenate((s.spectra, fullSpecs), axis=1)
            s.spectra[-1,-1,-1] = 0 # The last element is not needed here. It is the total number of attempts to generate spectra with SMARTS
            s.numSpectra[-1] = fullSpecs.shape[1]        
        
        s.Iscs = np.copy(s.spectra)
        s.P = np.zeros((2, s.spectra.shape[1]))
            
        def integra(d, specIndex): # Function to Integrate spectra from UV to given wavelength 
            s.P[d, specIndex] = integrate.trapz(s.spectra[d, specIndex, :], x=wavel) # Power per unit area ( W / m2 )
            s.Iscs[d, specIndex, :] = (q/hc)*np.insert(1e-9*integrate.cumtrapz(s.EQE*s.spectra[d, specIndex, :]*wavel, x=wavel), 0, 0) # Current per unit area ( A / m2 ), wavelength in nm
        for d in [0, 1]: # Integrate spectra from UV to given wavelength 
            for i in range(0, s.spectra.shape[1]):
                integra(d, i)
                
        if s.topJunctions==0:
            s.topJunctions = s.junctions
        if s.concentration>1:
            s.d = 1 # use direct spectra
        else:
            s.d = 0 # use global spectra
            s.coe = 1 # concentrator optical efficiency is irrelevant in this case
            
    def intSpec(s, energy, specIndex): 
        """ Returns integrated photocurrent from given photon energy to UV """
        return np.interp(1e9*hc/energy/q, wavel, s.Iscs[s.d, specIndex, :]) # interpolate integrated spectra. Wavelength in nm
        
    def getIjx(s, specIndex):
        """ Get cell T and external I in each junction, including EQE and junction transmission due to finite thickness """
        Ijx = np.zeros(s.junctions) # Array with the external photocurrents integrated from spectrum.
        IfromTop = 0
        upperIntSpec = 0
        s.T = s.Tmin + s.deltaT[ s.d ] * s.P[ s.d, specIndex ] / 1000 # To a first approximation, cell T is a linear function of irradiance.
        for i in range(s.junctions-1, -1, -1): # From top to bottom: get external photocurrent in each junction
            IntSpec = s.intSpec( s.gaps[i] + Varshni( s.T ), specIndex ) # IntSpec = Integrated current from UV to given Energy gap
            Ijx0 = s.concentration*(IntSpec-upperIntSpec) # Get external I per junction 
            upperIntSpec = IntSpec
            if i != 0:
                Ijx[i] = (1-s.transmission)*Ijx0+IfromTop # Subcell thickness cannot be infinite, 3 micron GaAs has transmission in the 2 to 3 % range (depending on integration range)
                IfromTop = s.transmission*Ijx0
            else:
                Ijx[i] = Ijx0+IfromTop # bottom junction does not transmit (back mirror)
        return Ijx
                
    def thin(s): # This function is normally not used. Only of interest in a few special cases with constrained band gaps. NOT FULLY TESTED
        """ Calculate transmission factors for each subcell to maximize current under spectrum given in .thinSpec """
        # Top cell thinning: 
        # - From top to bottom junction
        #       - If next current is lower:
        #             - Find average with next junction
        #             - If average I is larger than next junction I, extend average and repeat this step
        # Most general case considered is two mechanically stacked devices. Each device has 2 terminals. Stack has 4 terminals.
                
        Ijx = s.getIjx( s.thinSpec ) # get external photocurrent
        initialIjx = np.copy(Ijx)
        bottomJts = s.junctions-s.topJunctions # Bottom junction of the top stack
        stack = [ [s.junctions-1, bottomJts] ] # This is the top 2 terminal multijunction in the mechanical stack
        if bottomJts > 0: # If there are additional junctions
            stack.append([bottomJts-1, 0]) # Add the bottom 2 terminal multijunction in the mechanical stack
            
        for seriesDevice in stack: # First do top device in the mechanical stack
            topJ, bottomJ = seriesDevice
            Ijxmin = 0
            while int( Ijxmin * 100 ) < int( Ijx[ bottomJ : topJ + 1 ].min() * 100 ): # While min I keeps going up
                Ijxmin = Ijx[ bottomJ : topJ + 1 ].min()
                donorSubcell = topJ
                while donorSubcell > bottomJ: # Spread I from top to bottom
                    #for aceptorSubcell in range( donorSubcell-1, bottomJ-1, -1 ):
                    aceptorSubcell = donorSubcell - 1                    
                    if Ijx[ donorSubcell ] > Ijx[ aceptorSubcell ]: # If next current is lower, average current with lower junction to reduce excess
                        mean = np.mean( Ijx[ aceptorSubcell : donorSubcell + 1 ] )
                        previousMean = mean + 1                       
                        while mean < previousMean: # try to decrease excess current in mean by including lower subcells
                            previousMean = mean
                            aceptorSubcell -= 1 # include lower subcell in mean
                            if aceptorSubcell < bottomJ: # sanity check
                                aceptorSubcell = bottomJ
                            else:
                                mean = np.mean( Ijx[ aceptorSubcell : donorSubcell + 1 ] )
                        
                        if previousMean < mean: #backtrack if last attempt was failure
                            mean = previousMean
                            aceptorSubcell += 1
                                                        
                        Ijx[ aceptorSubcell : donorSubcell + 1] = mean # Change currents
                        donorSubcell = aceptorSubcell # jump over subcells that have been averaged
                    donorSubcell -= 1 # Go down 
                    
            if bottomJts > 0: # if there is a lower series device in stack
                exIb = Ijx[bottomJts] - Ijx[bottomJts:].min() # The excess current from the bottom junction of the top series, goes to lower series
                Ijx[bottomJts] -= exIb
                Ijx[bottomJts-1] += exIb    
                   
        s.thinTrans = Ijx / initialIjx

        #print ('in', initialIjx, initialIjx.sum())
        #print ('out', Ijx, Ijx.sum())
        #print (s.thinTrans)

    def timePerSpectra(s, numBins, binIndex):
        """ Returns fraction of yearly daytime represented by each spectra """
        if '.clusters.npy' in s.specsFile and numBins < 21:
            return s.timePerCluster[ s.d, getSpectrumIndex(numBins, binIndex) ] # Fraction of yearly daytime represented by each spectra
        else:
            return 1 / s.numSpectra[numBins] # Fraction of yearly daytime represented by each spectra. numSpectra = [1, 1, 2, 3, ...

    def series(s, topJ, bottomJ, numBins, binIndex):
        """ Get power from 2 terminal monolythic multijunction device. Series connected subcells with indexes topJ to bottomJ. topJ = bottomJ is single junction. """
        # 1 - Get external photocurrent in each junction
        # 2 - Get current at the maximum power point from min external photocurrent
        # 3 - Add radiative coupling, recalculate maximum power point, this changes radiative coupling, repeat until self consistency
        # 4 - Calculate power out
        
        # Do "tandems.show_assumptions()" to see EQE model used and some characteristics of the spectral set used.

        if (topJ<0):
            return
        Ijx = s.getIjx( getSpectrumIndex( numBins, binIndex ) ) * s.thinTrans # thinTrans is set by thin()
        # For a 1mm2 cell at 1000 suns bonded to copper substrate T = 70. Cite I. Garcia, in CPV Handbook, ed. by: I. Rey-Stolle, C. Algora
        kT = con.k*s.T
        k0 = 2 * np.pi * q * kT**3 / ( con.h * hc**2 )
        Irc0 = s.Irc #radiative coupling from upper stack 

        if Ijx.min() <=0:
            pdb.set_trace()
        Imax = Ijx[ bottomJ : topJ + 1 ].min() # Initial guess for current at the maximum power point 
        Imaxs = [0, Imax]
        while ( ( ( Imax-Imaxs[-2] ) / Imax )**2 ) > 1e-7: # Loop to self consistently refine max power point
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
                g0 = s.gaps[i] * q / kT
                #I0 = ( 1 + backLoss ) * 2 * np.pi * q * np.exp( -1 * s.gaps[i] * q / kT) * kT**3 * ( ( s.gaps[i] * q / kT + 1 )**2 + 1 ) / ( con.h * hc**2 ) / s.ERE
                I0 = ( 1 + backLoss ) * k0 * np.exp( -1 * g0 ) * ( ( g0 + 1 )**2 + 1 ) / s.ERE # Dark current at V = 0 in A / m2 s
                V += (kT/q)*np.log((Ij[i]-I)/I0+1) # add voltage of series connected cells
            V -= s.R*I # V drop due to series resistance
            Imax = I[np.argmax(I*V)] # I at max power point
            Imaxs.append(Imax)
        if len(Imaxs)>10:
            print ('self consistency is slowing convergence while finding the maximum power point.')
            print ('ERE or beta might be too high.')
            print ('Current at the maximum power point is converging as:', Imaxs)
            pdb.set_trace()
        s.Itotal += Ijmin * s.timePerSpectra( numBins, binIndex ) # Yearly daytime average
        s.Pout += (I*V).max() * s.timePerSpectra( numBins, binIndex ) # Yearly daytime average
        
    def stack(s, numBins, binIndex): # A stack is composed of one monolythic two terminal device or two of them mechanically stacked
        """ Use a single spectrum to get power from 2, 3, or 4 terminal tandem. If topJunctions = junctions the result is for 2 terminal tandem. """
        # Most general case considered is two mechanically stacked devices. Each device has 2 terminals. Stack has 4 terminals.
        s.Irc = 0 # For top cell there is no radiative coupling from upper cell
        s.series( s.junctions-1, s.junctions-s.topJunctions, numBins, binIndex ) # For top stack, calculate power out for each spectra and add to Pout. topJunctions is number of juntions in top stack
        if not s.opticallyCoupledStacks:
            s.Irc = 0
        s.series( s.junctions-s.topJunctions-1, 0, numBins, binIndex ) # For bottom stack, calculate power out for each spectra and add to Pout.
        return
            
    def useBins( s, bins, results ): 
        """Use spectral bins and s.gaps to calculate yearly energy yield and eff. Optionally discard bad results."""           
        for numBins in bins: # loop for number of bins
            Pin = 0
            s.Itotal = 0
            s.Pout = 0
            for binIndex in range(s.numSpectra[numBins]-1, -1, -1): # loop for bin index.    numSpectra = [1, 1, 2, 3, ...
                s.stack( numBins, binIndex ) # Calculate power in/out for each spectra and add to Pin and Pout
                Pin += s.P[ s.d, getSpectrumIndex( numBins, binIndex ) ] * s.timePerSpectra( numBins, binIndex ) # Yearly averaged daytime power in from the sun
            eff = s.Pout / Pin / s.concentration
            s.effs[results, numBins] = eff
            s.Is[results, numBins] = s.Itotal
        s.rgaps[results,:] = s.gaps
        s.auxEffs[results, :] = np.zeros(s.junctions)+s.effs[results, numBins]
        s.auxIs[results, :] = np.zeros(s.junctions)+s.Itotal
        if eff < ( s.effs[:, numBins].max() - s.effMin ):  # Will overwrite this result if it is not good enough
            return results
        return results + 1

    def findGaps(s): 
        """ Calculate efficiencies for random band gap combinations. """
        startTime = time.time()
        ncells = 0 # Number of calculated gap combinations
        results = 0
        effmax = 0
        Emin_ = np.array(Emin[s.junctions-1]) # Initial guess at where the eff maxima are. 
        Emax_ = np.array(Emax[s.junctions-1]) # The search space is expanded as needed if high eff are found at the edges of the search space.        
        minDif_ = np.array(minDif[s.junctions-2])
        maxDif_ = np.array(maxDif[s.junctions-2])
        # REMOVE SEED unless you want to reuse the same sequence of random numbers (e.g.: compare results after changing one parameter)
        #np.random.seed(07022015)
        s.rgaps = np.zeros((s.cells, s.junctions)) # Gaps
        s.auxIs = np.zeros((s.cells, s.junctions)) # Aux array for plotting only
        s.auxEffs = np.zeros((s.cells, s.junctions)) # Aux array for plotting only  
        s.Is = np.zeros((s.cells, 22)) # Currents as a function of the number of spectral bins, 0 is standard spectrum, 10 is full spectral dataset
        s.effs = np.zeros((s.cells, 22)) # Efficiencies as a function of the number of spectral bins, 0 is standard spectrum, 10 is full spectral dataset
        fixedGaps = np.copy(s.gaps) # Copy input gaps to remember which ones are fixed, if gap==0 make it random
        while (results<s.cells): # Loop to randomly sample a large number of gap combinations
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
                s.thin()
            
            s.useBins( [2], results )
            eff = s.effs[results, 2] # get rough estimate of efficiency
            
            if ( eff > effmax - s.effMin - 0.01 ): # If gap combination is good, do more work on it
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
                results = s.useBins( s.bins, results ) # Calculate efficiency for s.gaps, store result if gaps are good
                if results % 10 == 0: # Show calculation progress from time to time
                    print ('Tried '+str(ncells)+', got '+str(results)+' candidate gap combinations.', end="\r")                                
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
        s.effs = np.reshape(s.effs, (-1, 22))
        s.Is = np.reshape(s.Is, (-1, 22))
        
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
        
    def plot(s, dotSize=1, show=True):
        """ Saves efficiency plots to PNG files """ 
        res = np.size(s.rgaps)/s.junctions
        Is_ = np.copy(s.auxIs)
        Is_ = (Is_-Is_.min())/(Is_.max()-Is_.min())
        if s.convergence:
            plt.figure()
            plt.xlabel('Number of spectra')
            plt.ylabel('Yearly efficiency overestimate (%)')
            plt.xticks(list(range(1,21)))
            plt.xlim(1,20)
            plt.ylim( 0, 1.01*( 100 * ( s.effs[ : , 1 ] - s.effs[ : , -1 ] ) ).max() )  
            plt.tick_params(axis='y', right='on')
            for i in range(0, int(res)):
                plt.plot( range(1,21), 100*( s.effs[ i , 1:-1 ] - s.effs[ i , -1 ] ), color = LGBT(Is_[i,0]), linewidth=0.5) #  color = LGBT(auxIs[i*s.junctions])
            plt.savefig(s.name+' '+s.specsFile.replace('.npy', '').replace('.','-')+' convergence '+str(int(s.junctions))+' '+str(int(s.topJunctions))+' '+str(int(s.concentration))+' '+s.autoSuffix, dpi=600)
            if show:
                plt.show()
        plt.figure()
        ymin = (s.effs[:, s.bins[-1]].max()-s.effMin)
        ymax = s.effs[:, s.bins[-1]].max() + 0.001
        plt.ylim(100*ymin, 100*ymax)
        plt.xlim(0.5, 2.5)
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
        plt.scatter( s.rgaps, 100 * s.auxEffs, c=Is_, s = dotSize * 100000 / len( s.effs[ :, s.bins[ -1 ] ] ), edgecolor='none', cmap=LGBT)
        axb = plt.gca().twinx()
        axb.set_ylim(s.kWh(ymin),s.kWh(ymax))
        axb.set_ylabel('Yield $\mathregular{(kWh / m^2 / year)}$')
        plt.savefig(s.name+' '+s.specsFile.replace('.npy', '').replace('.','-')+' '+str(int(s.junctions))+' '+str(int(s.topJunctions))+' '+str(int(s.concentration))+' '+s.autoSuffix, dpi=600)
        if show:
            plt.show()
        plt.figure()
        plt.xlim(100*(s.auxEffs.max()-s.effMin), 100*s.auxEffs.max())
        plt.xlabel('Yearly averaged efficiency (%)')
        plt.ylabel('Count')
        plt.hist(100*s.effs[:, s.bins[-1]], bins=30)
        plt.savefig(s.name+' Hist '+s.specsFile.replace('.npy', '').replace('.','-')+' '+str(int(s.junctions))+' '+str(int(s.topJunctions))+' '+str(int(s.concentration))+' '+s.autoSuffix, dpi=600)      
        if show:
            plt.show()  
        
    def save(s, saveSpectra=False):
        """ Saves data for later reuse/replotting. Path and file name set in eff.name, some parameters and autoSuffix are appended to filename """
        s2 = s
        if not saveSpectra:
            s2 = deepcopy(s)
            s2.spectra = 0
            s2.Iscs = 0
        with open(s.name+' '+s.specsFile.replace('.npy', '')+' '+str(int(s.junctions))+' '+str(int(s.topJunctions))+' '+str(int(s.concentration))+' '+s.autoSuffix, "wb") as f:
            f.write( json_tricks.dumps( s2, compression=True ) )
        
    def recalculate(s):
        """ Recalculate efficiencies with new set of input parameters using previously found gaps. Call __init__() to change parameters before recalculate() """
        for gi in range( 0, s.rgaps.shape[0] ):
            s.gaps = s.rgaps[gi]
            s.useBins( s.bins, gi ) # Calculate efficiency for s.gaps
        s.results()
        
    def compare( s, s0, dotSize=1, show=True): 
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
        plt.scatter(s0.auxIs , diff , c=Is_ , s=dotSize*20000/len(s.effs[:, s.bins[-1]]) , edgecolor='none', cmap=LGBT)
        plt.savefig(s.name+'-'+s0.name.replace('/', '').replace('.', '')+' '+s.specsFile.replace('.npy', '').replace('/', '').replace('.', '_')+'-'+s0.specsFile.replace('.npy', '').replace('/', '').replace('.','_')+' '+str(int(s0.junctions))+' '+str(int(s0.topJunctions))+' '+str(int(s0.concentration))+' '+s.autoSuffix, dpi=600)
        if show:
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
        speCount = 20000, NSRDBfile='', fname='spectra', saveFullSpectra = False, loadFullSpectra = False,
        method='clusters', numFeatures = 200    ): # if method is set to anything else, Garcia's binning method will be used. DOI: 10.1002/pip.2943  
    
    def EPR(i, s1): 
        """ Calculates EPR, integrates Power and stores spectra"""
        P[:,i] = integrate.trapz(s1, x=wavel, axis=1)
        if (P[0, i] > 0) and (P[1, i] > 0):
            fullSpectra[:, i, :] = s1
            EPR650[:, i] = integrate.trapz( s1[:, :490], x=wavel[:490], axis=1 )
            EPR650[:, i] = ( P[:, i] - EPR650[:, i] ) / EPR650[:, i] # Calculate EPR, Ivan Garcia's criteria for binning
            i += 1 # Spectrum index is incremented if spectrum is not zero
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

    fullSpectra = np.zeros((2, speCount, len(wavel))) # Array to hold generated spectra, first index is 0=global or 1=direct
    EPR650 = np.zeros((2, speCount)) # Ivan Garcia's criteria for binning, store EPR value for each spectrum
    P = np.zeros((2, speCount)) # Power for each spectrum

    if NSRDBfile != '': # Optionally Load specified National Solar Resource DataBase spectra
        # https://maps.nrel.gov 
        Ng = np.loadtxt(NSRDBfile, skiprows=3, delimiter=',', usecols=tuple(range(159, 310))) # Global Horizontal
        Nd = np.loadtxt(NSRDBfile, skiprows=3, delimiter=',', usecols=tuple(range(8, 159))) # Direct spectra
        attempts = Nd.shape[0]
        if fname == 'spectra':
            fname = NSRDBfile.replace('.csv', '')
        s1 = np.zeros((2, len(wavel)))
        speCount = 0
        for i in range(0, len(Nd)):
            s1[0,:] = np.interp(wavel, np.arange(300, 1810, 10), Ng[i, :], left=0, right=0) # Global Horizontal
            s1[1,:] = np.interp(wavel, np.arange(300, 1810, 10), Nd[i, :], left=0, right=0) # Direct spectra
            speCount = EPR(speCount, s1) # Calculates EPR, integrates Power and stores spectra
            print (speCount, 'spectra out of', attempts, 'points in time', end="\r")
        fullSpectra = fullSpectra[ :, :speCount, : ] # Array to hold the whole set of spectra
        EPR650 = EPR650[ :, :speCount ] # Ivan Garcia's criteria for binning, store EPR value for each spectrum
        P = P[:,:speCount] # Power for each spectrum        

    elif loadFullSpectra: # To reuse a full set of spectra saved earlier 
        fullSpectra = np.load(fname+'.full.npy')
        attempts = fullSpectra[-1,-1,-1]
        fullSpectra[-1,-1,-1] = 0
        
        for specIndex in range(0, speCount):
            EPR(specIndex, fullSpectra[:,specIndex,:]) # Calculates EPR, integrates Power and stores spectra

    else:# Generate SMARTS spectra if no file with spectra is provided
        longitude2 = longitude
        AOD2 = AOD
        PW2 = PW
        #os.chdir('/path to your SMARTS files')
        with open ('smarts295.in_.txt', "r") as fil: # Load template for SMARTS input file
            t0 = fil.readlines()
        t0 = ''.join(t0)
        attempts = 0
        t0 = t0.replace('38 -999 -999', tracking)
        for specIndex in range(0, speCount):
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
                sub.check_call('./smartsAM4', shell=True) # execute SMARTS
                tama = os.stat('smarts295.ext.txt').st_size # check output file
                if tama>0:
                    try: # load output file if it exists
                        s1 = np.loadtxt("smarts295.ext.txt", delimiter=' ', usecols=(0, 1, 2), unpack=True, skiprows=1)#Global Tilted = 1, Direct Normal= 2, Diffuse Horizontal=3
                    except:
                        tama = 0
            EPR(specIndex, s1[1:,:]) # Calculates EPR, integrates Power and stores spectra
        print ('Finished generating spectra. Called Smarts '+str(attempts)+' times and got '+str(speCount)+' spectra')
    
    if saveFullSpectra:
        fullSpectra[-1,-1,-1] = attempts
        np.save(fname+'.full', fullSpectra)
    
    totalPower = np.zeros(2)
    binlimits = []
    binspecs = []
    spectra = np.zeros( (2, 211, len(wavel)) ) # Prepare data structure for saving to file
    timePerCluster = np.zeros( (2, 211) ) # Each spectrum is representative of this fraction of yearly daytime
    
    if method == 'clusters': # Use machine learning clustering methods            
        # Step 1: Search for features in the spectra that show the same trends as a function of time       
        conn = np.zeros( ( len(wavel), len(wavel) ), dtype=int) # Connectivity matrix for spectral feature identification
        for i in range(1, len(wavel)-1): # Only allow adjacent points to be in the same spectral feature
            conn[i, i] = 1
            conn[i, i+1] = 1
            conn[i, i-1] = 1
        conn[0,0] = 1
        conn[0,1] = 1
        conn[len(wavel)-1,len(wavel)-1] = 1
        conn[len(wavel)-1,len(wavel)-2] = 1
        
        specFeat = np.zeros((2, speCount, numFeatures)) 
             
        for d in (0,1): # d = 1 for direct spectra
            # Begin by searching for characteristic spectral features
            features = FeatureAgglomeration(n_clusters=numFeatures, connectivity=conn).fit(fullSpectra[d,:,:])   
                    
            for fea in range(numFeatures):
                mask = features.labels_ == fea
                specFeat[d, :, fea] = fullSpectra[d, :, mask].mean(axis=0)
                
            # Now merge those spectra that have  nearly the same intensity for their characteristic features      
            specIndex = 1
            for numBins in range(1, 21): # number of bins

                #clusters = AgglomerativeClustering(n_clusters=numBins, linkage='ward').fit(specFeat[d,:,:]) # Find clusters of spectra
                clusters = KMeans(n_clusters=numBins).fit(specFeat[d,:,:]) # Find clusters of spectra
                
                for binIndex in range(0, numBins): 
                    mask = clusters.labels_ == binIndex
                    spectra[d, specIndex, :] = fullSpectra[d,mask,:].mean(axis=0)
                    timePerCluster[d, specIndex] = mask.sum()/speCount
                    specIndex += 1
        
        np.save(fname+'.timePerCluster', timePerCluster)
        fname += '.clusters'
        
    else: # If not using machine learning clusters, use Garcia's binning method based on the EPR650 criteria, DOI: 10.1002/pip.2943        
        for d in (0,1): # d = 1 for direct spectra
            fullSpectra[d, :, :] = fullSpectra[ d, np.argsort(EPR650[d,:]), : ] # Sort spectra and integrated powers by EPR
            P[d, :] = P[d, np.argsort(EPR650[d,:])]
            totalPower[d] = P[d, :].sum() # Sum power
            binlimits.append([])
            binspecs.append([])
            binIndex = np.zeros(21, dtype=np.int)
            for numBins in range(1, 21): # Prepare data structure to store bin limits and averaged spectra. numBins is total number of bins.
                binlimits[d].append( np.zeros( numBins+1, dtype=np.int ) )
                binspecs[d].append( np.zeros( ( numBins, len(wavel) ) ) )
            accubin = 0 
            for specIndex in range(0, speCount): # Calculate bin limits with equal power in each bin
                accubin += P[d, specIndex] # Accumulate power and check if its a fraction of total power
                for numBins in range(1, 21):
                    if accubin >= ( binIndex[ numBins ] + 1 ) * totalPower[d] / numBins: # check if power accumulated is a fraction of total power
                        binIndex[ numBins ] += 1 # Create new bin if it is
                        binlimits[d][ numBins - 1 ][ binIndex[ numBins ] ] = specIndex+1 # Store bin limit     
            for numBins in range(1, 21): #iterate over every bin set
                binlimits[d][numBins-1][-1] = speCount # set the last bin limit to the total number of spectra
            # Average spectra using the previously calculated bin limits
            for numBins in range(1, 21):
                for binIndex in range(0, numBins): 
                    binspecs[d][numBins-1][binIndex,:] = np.mean( fullSpectra[d, binlimits[d][numBins-1][binIndex]:binlimits[d][numBins-1][binIndex+1]], axis=0 ) # mean each bin
            specIndex = 1
            for numBins in range(1, 21): # number of bins
                for binIndex in range(0, numBins): #bin counter
                    spectra[d, specIndex, :] = binspecs[d][numBins-1][binIndex,:]
                    specIndex += 1
        fname += '.bins'
    
    spectra[0, 0, -1] = speCount/attempts # This number is needed to calculate the yearly averaged power yield including night time hours. 
    np.save(fname, spectra)    
    
def docs(): # Shows HELP file
    with open('HELP', 'r') as fin:
        print (fin.read())
        
