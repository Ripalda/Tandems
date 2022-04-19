# Yearly averaged photovoltaic efficiencies.
# SMARTS 2.9.5 and sklearn are required only to generate new sets of spectra.
# Provided files with ".npy" or ".npz" extension can be used instead to load a set of spectra.
# Clone or download from https://github.com/Ripalda/Tandems to obtain full set of spectra (about 600 MB).

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time
import pdb
import pickle
from copy import deepcopy
from datetime import datetime
from glob import glob
from sklearn.cluster import FeatureAgglomeration
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import normalize
from scipy import integrate
import scipy.constants as con
import os

__author__ = 'Jose M. Ripalda'
__version__ = 0.989

print('Tandems version', __version__)

Dpath = os.path.dirname(__file__)  # Data files here
if Dpath == "":
    Dpath = "data/"
else:
    Dpath = Dpath + "/data/"
print('Data path is: ', Dpath)

hc = con.h*con.c
q = con.e

# Make Numpy print 3 decimal places only
np.set_printoptions(precision=3)

# Define color scale for plots
colors = [[ 0.3 ,  0.  ,  0.2 ],
   [ 0.5 ,  0.  ,  0.35],
   [ 0.7 ,  0.  ,  0.5 ],
   [ 0.35,  0.  ,  0.75],
   [ 0.  ,  0.  ,  1.  ],
   [ 0.  ,  0.25,  0.75],
   [ 0.  ,  0.5 ,  0.5 ],
   [ 0.  ,  0.75,  0.75],
   [ 0.  ,  0.9  ,  0.9  ],
   [ 0.  ,  1  ,  0.  ],
   [ 0.4 ,  1 ,  0.  ],
   [ 0.85,  0.9 ,  0.  ],
   [ 1.  ,  0.9 ,  0.  ],
   [ 1.  ,  0.7 ,  0.1 ],
   [ 1.  ,  0.5 ,  0.2 ],
   [ 1.  ,  0.25,  0.1 ],
   [ 1.  ,  0.  ,  0.  ],
   [ 1.  ,  0.3 ,  0.4 ],
   [ 1.  ,  0.6 ,  0.8 ]]

LGBT = LinearSegmentedColormap.from_list('LGBT', colors, 500)

# Load standard reference spectra ASTM G173
wavel, g1_5, d1_5 = np.loadtxt(Dpath + "AM1_5 smarts295.ext.txt",
                               delimiter=',', usecols=(0, 1, 2), unpack=True)
Energies = 1e9*hc/q/wavel  # wavel is wavelenght in nm
binMax = 31  # Maximum number of clusters (or bins) + 1
# 2D Array for spectral bin indexing.
# First index is number of bins, second is bin index.
specIndexArray = []
specIndexArray.append([0])
specIndex = 1
for numBins in range(1, binMax):  # bin set
    specIndexArray.append([])
    for binIndex in range(0, numBins):  # bin counter
        specIndexArray[numBins].append(specIndex)
        specIndex += 1
specMax = specIndex


# Spectrum index for input in spectra array.
def getSpectrumIndex(numBins, binIndex):
    if numBins >= binMax:  # Use full set of spectra
        return binIndex + specMax
    else:  # Use averaged proxy spectra
        return specIndexArray[numBins][binIndex]


def load_np(filename):  # Convinience wraper on numpy.load
    loaded_array = []
    if '.npy' in filename:
        loaded_array = np.load(filename)
    elif '.npz' in filename:
        temp_zip = np.load(filename)
        loaded_array = temp_zip['arr_0']
    else:
        print('File must be .npy or .npz')
    return loaded_array

def Varshni(T):
    """
    Input parameter:
        T (float): Temperature in K
    returns:
        float : the band gap energy correction relative to 300K.

    Gives gap correction in eV relative to 300K using GaAs parameters. T in K
    Beware, using GaAs parameters slightly overestimates the effect for most
    other semiconductors

    Varshni, Y. P. Temperature dependence of the energy gap in
    semiconductors.  Physica 34, 149 (1967)
    """
    return (T**2)/(T+572)*-8.871e-4+0.091558486

def sandia_T(poa_global, wind_speed, temp_air):
    """ Sandia solar cell temperature model
    Adapted from pvlib library to avoid using pandas dataframes
    parameters used are those of 'open_rack_cell_polymerback'
    """

    a = -3.56
    b = -0.075
    deltaT = 3

    E0 = 1000.  # Reference irradiance

    temp_module = poa_global * np.exp(a + b * wind_speed) + temp_air

    temp_cell = temp_module + (poa_global / E0) * (deltaT)

    return temp_cell

def physicaliam(aoi, n=1.526, K=4., L=0.002):
    '''
    Determine the incidence angle modifier using refractive index,
    extinction coefficient, and glazing thickness.

    Adapted from pvlib library to avoid using pandas dataframes

    physicaliam calculates the incidence angle modifier as described in
    De Soto et al. "Improvement and validation of a model for
    photovoltaic array performance", section 3. The calculation is based
    on a physical model of absorbtion and transmission through a
    cover.

    Note: The authors of this function believe that eqn. 14 in [1] is
    incorrect. This function uses the following equation in its place:
    theta_r = arcsin(1/n * sin(aoi))

    Parameters
    ----------
    aoi : numeric
        The angle of incidence between the module normal vector and the
        sun-beam vector in degrees. Angles of 0 are replaced with 1e-06
        to ensure non-nan results. Angles of nan will result in nan.

    n : numeric, default 1.526
        The effective index of refraction (unitless). Reference [1]
        indicates that a value of 1.526 is acceptable for glass. n must
        be a numeric scalar or vector with all values >=0. If n is a
        vector, it must be the same size as all other input vectors.

    K : numeric, default 4.0
        The glazing extinction coefficient in units of 1/meters.
        Reference [1] indicates that a value of  4 is reasonable for
        "water white" glass. K must be a numeric scalar or vector with
        all values >=0. If K is a vector, it must be the same size as
        all other input vectors.

    L : numeric, default 0.002
        The glazing thickness in units of meters. Reference [1]
        indicates that 0.002 meters (2 mm) is reasonable for most
        glass-covered PV panels. L must be a numeric scalar or vector
        with all values >=0. If L is a vector, it must be the same size
        as all other input vectors.

    Returns
    -------
    iam : numeric
        The incident angle modifier

    References
    ----------
    [1] W. De Soto et al., "Improvement and validation of a model for
    photovoltaic array performance", Solar Energy, vol 80, pp. 78-88,
    2006.

    [2] Duffie, John A. & Beckman, William A.. (2006). Solar Engineering
    of Thermal Processes, third edition. [Books24x7 version] Available
    from http://common.books24x7.com/toc.aspx?bookid=17160.

    See Also
    --------
    getaoi
    ephemeris
    spa
    ashraeiam
    '''
    if aoi < 1e-6:
        iam = 1
    elif aoi > 89.99:
        iam = 0
    else:

        # angle of reflection
        thetar_deg = np.degrees(np.arcsin(1.0 / n * (np.sin(np.radians(aoi)))))

        # reflectance and transmittance for normal incidence light
        rho_zero = ((1-n) / (1+n)) ** 2
        tau_zero = np.exp(-K*L)

        # reflectance for parallel and perpendicular polarized light
        rho_para = (np.tan(np.radians(thetar_deg - aoi)) /
                    np.tan(np.radians(thetar_deg + aoi))) ** 2
        rho_perp = (np.sin(np.radians(thetar_deg - aoi)) /
                    np.sin(np.radians(thetar_deg + aoi))) ** 2

        # transmittance for non-normal light
        tau = np.exp(-K*L / np.cos(np.radians(thetar_deg)))

        # iam is ratio of non-normal to normal incidence transmitted light
        # after deducting the reflected portion of each
        iam = ((1 - (rho_para + rho_perp) / 2) / (1 - rho_zero) * tau / tau_zero)

    return iam


class effs(object):
    """ Object class to hold results sets of yearly average photovoltaic efficiency

    ---- USAGE EXAMPLE ----

    import tandems

    tandems.docs()    # Shows help file

    #    Include as many or as few options as needed.
    eff = tandems.effs(junctions=4, bins=6, concentration=500)
    eff.findGaps() #  Find set of optimal gap combinations.
    eff.plot() #  Figures saved to PNG files.

    eff.save() # Data saved for later reuse/replotting.
    Path and file name set in eff.name,
    some parameters and autoSuffix are appended to filename

    eff2 = tandems.copy.deepcopy(eff)
    eff2.__init__(bins=8, concentration=1, R=4e-5)
    #  Change some input parameters but keep previously
    found set of optimal gap combinations.

    eff2.recalculate() #  Recalculate efficiencies for
    #  previously found set of optimal gap combinations.
    eff2.compare(eff) # Compares efficiencies in two datasets by
    #  doing eff2 - eff. Plots difference and saves PNG files.

    eff = tandems.load('/path/and file name here') # Load previusly saved data
    eff.results()
    eff.plot()
    """

    # s = self = current object instance
    # ---- Input variables ----
    junctions = 6
    topJunctions = 0  # Number of series conected juctions in top stack (topJunctions = 0 in 2 terminal devices)
    concentration = 1000
    gaps = [0, 0, 0, 0, 0, 0]  # First is bottom. If a gap is 0, it is randomly chosen by tandems.findGaps(), otherwise it is kept fixed at value given here.
    ERE = 0.01  # External radiative efficiency without mirror. With mirror ERE increases by a factor (1 + beta)
    beta = 11  # n^2 squared refractive index  =  radiative coupling parameter  =  substrate loss.
    bins = 15  # bins is number of spectra used to evaluate eff. See convergence = True.
    #bins = np.arange(1,binMax,1) # This is to test convergence as a function of the number of bins or clusters
    T_from_spec_file = False  # Use cell temperature data embeded in the spectra
    expected_eff = 0.3  # Expected yearly averaged efficiency, this is only used to calculate the solar cell temperature if T_from_spec_file = True
    Tmin = 15+273.15  # Minimum cell temperature at zero irradiance in K. Only used if T_from_spec_file = False
    deltaT = np.array([30, 55])  # Device T increase over Tmin caused by high irradiance (1000 W/m2), first value is for one sun cell, second for high concentration cell. Only used if T_from_spec_file = False
    convergence = False  # Set to True to test the effect of changing the number of spectral bins used to calculate the yearly average efficiency
    transmission = 0.02  # Subcell thickness cannot be infinite, 3 micron GaAs has transmission in the 2 to 3 % range (depending on integration range)
    thinning = False  # Automatic subcell thinning for current matching
    thinSpec = 1  # Spectrum used to calculate subcell thinning for current matching. Integer index in spectra array.
    effMin = 0.02  # Lowest sampled efficiency value relative to maximum efficiency. Gaps with lower efficiency are discarded.
    d = 1  # 0 for global spectra, 1 for direct spectra
    # T = 70 for a 1mm2 cell at 1000 suns bonded to copper substrate. Cite I. Garcia et al., in CPV Handbook, ed. by: I. Rey-Stolle, C. Algora
    name = './Test '  # Can optionally include path to destination of generated files. Example: "/home/documents/test". Some parameters and autoSuffix are appended to filename
    cells = 1000  # Target number of calculated tandem cells. Actual number of results will be less.
    R = 5e-7  # Series resistance of each stack in Ohm*m2. Default is optimistic value for high concentration devices
    # R = 4e-5 is suggested as optimistic value for one sun flat plate devices
    EQE = 0.7453*np.exp(-((Energies-1.782)/1.384)**4)+0.1992  # EQE model fitted to current record device, DOI.: 10.1109/JPHOTOV.2015.2501729
    mirrorLoss = 1  # Default value = 1 implies the assumption that back mirror loss = loss due to an air gap.
    # A group of series connected junctions is here called a stack. Up to two mechanically stacked, optically coupled, but electrically independent "stacks" are considered
    opticallyCoupledStacks = False  # Bottom junction of the top terminal stack can either have photon recycling or radiative coupling to the botttom stack.
    coe = 0.9  # Concentrator optical efficiency. Optimistic default value. Used only for yield calculation.
    cloudCover = 0 # Fraction of the yearly energy that is lost due to clouds if clear sky spectra are being used.
    # Used only for yield calculation. A value of 0.26 for direct spectra is representative of area near Denver, CO. and is recommended if clear sky spectra (such as those generated by SMARTS) are being used.
    # A slightly smaller cloudCover (0.25 vs 0.26) is recommended for global spectra.
    # IMPORTANT set cloudCover = 0 when using experimental spectra or spectra obtained from a model including the effect of clouds.
    specsFile = 'lat40.clusters.npy'  # Name of the file with the spectral set obtained from tandems.generate_spectral_bins(). See genBins.py
    # File types for spectral sets should either be .clusters.npy or .bins.npy


    # ---- Results ----

    rgaps = np.zeros((1, 6))  # Results Array with high efficiency Gap combinations found by trial and error
    Is = np.zeros((1, binMax+1))  # Results Array with Currents as a function of the number of spectral bins, 0 is standard spectrum
    effs = np.zeros((1, binMax+1))  # Results Array with Efficiencies as a function of the number of spectral bins, 0 is standard spectrum
    Pbins = np.zeros(binMax+1)  # Power per bin

    # ---- Internal variables ----

    Irc = 0  # Radiative coupling current
    Itotal = 0  # Yearly average (daytime only) photocurrent per unit cell area
    Pout = 0  # Yearly average (daytime only) power per unit module area
    T = 300  # Set from irradiance at run time in Kelvin
    auxEffs = np.zeros((1, 6))  # Aux array for efficiencies. Has the same shape as rgaps for plotting and array masking.
    auxIs = np.zeros((1, 6))  # Aux array for plotting. sum of short circuit currents from all terminals.
    spectra = []  # Spectral set loaded from file
    P = []  # Array with integrated power in each spectrum
    Iscs = []  # Array with current in each spectrum
    thinTrans = 1  # Array with transmission of each subcell
    autoSuffix = ''
    daytimeFraction = 1  # Needed to calculate the yearly averaged power yield including night time hours.
    numSpectra = [1]+list(range(1, binMax))+[10000]  # Number of spectra as a function of numBins
    timePerCluster = 0  # Fraction of the yearly daytime hours that is represented by each cluster of spectra. Set in __init__
    filenameSuffix = ''  # Part of automatic filenaming

    # ---- Methods and functions ----
    # s.findGaps() generates random band gap combinations, calls s.useBins() to calculate efficiencies and s.results() for output.
    # s.useBins() calculate averaged efficiency using spectral bins (or clusters), calls s.stack()
    # s.stack() get power from 2, 3, or 4 terminal tandem from a single spectrum, calls s.series()
    # s.series() get power from a single set of series conected junctions, calls s.getIjx()
    # s.getIjx() get cell T and external photocurrent at each subcell, calls s.intSpec()
    # s.intSpec() returns integrated photocurrent from given photon energy to UV, uses s.Iscs[] (integrated spectra) set by s.__init__()

    def __init__(s, **kwargs):
        """ Call __init__ if you want to change input parameters without discarding
        previously found gaps before calling recalculate()
        """

        # This is for having an automatic unique suffix in filenames
        # to reduce the chance of accidental overwriting
        alphabet = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        number = int((62**6-1)*np.random.rand())
        while number != 0:
            number, i = divmod(number, 62)
            s.autoSuffix = alphabet[i] + s.autoSuffix

        # This sets class attributes from input
        for k, v in kwargs.items():
            getattr(s, k)  # This is here so an error warns you if you mistype an attribute name
            setattr(s, k, v)

        if type(s.bins) == int:
            s.bins = [s.bins]

        # If you just want to calculate the efficiency of a single specific band gap combination, things need to be set up a bit differently
        if s.cells == 1:
            s.rgaps = np.array([s.gaps])
            s.auxIs = np.zeros((s.cells, s.junctions))  # Aux array for plotting only
            s.auxEffs = np.zeros((s.cells, s.junctions))  # Aux array for plotting only
            # Currents as a function of the number of spectral bins,
            # 0 is standard spectrum, binMax is full spectral dataset
            s.Is = np.zeros((s.cells, binMax + 1))
            # Efficiencies as a function of the number of spectral bins,
            s.effs = np.zeros((s.cells, binMax + 1))

        s.spectra = load_np(Dpath + s.specsFile)  # Load binned or clustered spectra
        # s.daytimeFraction is the fraction of daytime hours in a year and is
        # Needed to calculate the yearly averaged power yield including night time hours.
        s.daytimeFraction = s.spectra[0, 0, -1]  # It is hidden here when the spectra are loaded.
        s.spectra[0, 0, -1] = 0
        # These are for the standard reference spectra ASTM G173.
        s.spectra[0, 0, :] = g1_5
        s.spectra[1, 0, :] = d1_5

        if '.clusters.np' in s.specsFile:
            s.timePerCluster = load_np(Dpath + s.specsFile.replace('.clusters.', '.timePerCluster.'))
        elif '.bins.np' in s.specsFile:
            s.timePerCluster = load_np(Dpath + s.specsFile.replace('.bins.', '.timePerBin.'))
        else:
            print('File types for spectral sets should one of .clusters.npy, .bins.npy, .clusters.npz, or .bins.npz')
            return
        # These are for the standard reference spectra ASTM G173.
        s.timePerCluster[0, 0] = 1
        s.timePerCluster[1, 0] = 1

        if False:  # Set to True only for comparison with TMYSPEC spectra
            s.spectra[:, :, 1562:] = 0
            print('s.spectra[:, :, 1562:] = 0')
            print('Spectra are being cut short so efficiencies are comparable with eff from TMYSPEC spectra')
            print('Maximum wavelength is', wavel[int(s.spectra.shape[-1])-1], 'nm')

        if s.convergence:
            lastbin = s.bins[-1]
            s.bins = s.bins[:-1] + [binMax, lastbin]  # bin == binMax is the full set of spectra, bin 1 is the average
            fullSpecs = load_np(Dpath + s.specsFile.replace('.clusters.np',
                '.full.np').replace('.bins.np', '.full.np'))

            s.spectra = np.concatenate((s.spectra, fullSpecs), axis=1)
            s.spectra[-1, -1, -1] = 0
            # The last element is not needed here.
            # It's the total number of attempts to generate spectra with SMARTS
            s.numSpectra[-1] = fullSpecs.shape[1]
            s.Pbins = np.zeros(s.numSpectra[-1])

        s.spectra = np.nan_to_num(s.spectra)
        s.spectra[s.spectra < 0] = 0  # Negative values and NAN not allowed

        s.Iscs = np.copy(s.spectra)
        s.spectra_copy = np.copy(s.spectra)
        s.P = np.zeros((2, s.spectra.shape[1]))

        # Integrate spectra from UV to given wavelength.
        for d in [0, 1]:
            for specIndex in range(0, s.spectra.shape[1]):
                s.spectra_copy[d, :, :6] = 0  # The first 5 elements are sometimes used to store other data such as temperatures and wind speed
                # Power per unit area ( W / m2 )
                s.P[d, specIndex] = integrate.trapz(s.spectra_copy[d, specIndex, :], x=wavel)
                # Current per unit area ( A / m2 ), wavelength in nm
                s.Iscs[d, specIndex, :] = (q/hc)*np.insert(1e-9*integrate.cumtrapz(
                        s.EQE * s.spectra_copy[d, specIndex, :] * wavel, x=wavel), 0, 0)

        if s.topJunctions == 0:
            s.topJunctions = s.junctions
        if s.concentration > 1:
            s.d = 1  # use direct spectra
        else:
            s.d = 0  # use global spectra

        s.filenameSuffix = str(int(s.junctions))+' '+str(int(s.topJunctions))+' '+str(int(s.concentration))+' '+s.autoSuffix

        for i, bin in enumerate(s.bins):
            if bin < 0:  # This is used to calculate the efficiency with the standard spectrum but with the actual irradiance
                 s.bins[i] = -1 * s.bins[i]
                 for binIndex in range(s.numSpectra[s.bins[i]]-1, -1, -1):
                    spec_index =  getSpectrumIndex(s.bins[i], binIndex)
                    s.Iscs[0, spec_index, :] = s.Iscs[0, 0, :] * s.P[0, spec_index] / s.P[0, 0] # Only implemented for global spectrum TODO: perhaps change the whole code to hold a single set of spectra, either direct or global

    def intSpec(s, energy, specIndex):
        """ Interpolate integrated spectra. Wavelength in nm
        Returns integrated photocurrent from given photon energy to UV
        """
        return np.interp(1e9*hc/energy/q, wavel, s.Iscs[s.d, specIndex, :])

    def heat_eff(s, R=0.05, eff0=0.165):
        """Irradiance modifier to account for the fact that the Sandia cell
        temperature model does not include the effect of efficiency
        """
        return (1 - R - s.expected_eff) / (1 - R - eff0)

    def getIjx(s, specIndex):
        """ Get cell T and external I in each junction at 1 sun,
        including EQE and junction transmission due to finite thickness
        """
        Ijx = np.zeros(s.junctions)  # Array with the external photocurrents
        IfromTop = 0
        upperIntSpec = 0

        if s.T_from_spec_file:  # Find solar cell temperature
            amb_T = s.spectra[s.d, specIndex, 1] * 1e6
            wind = s.spectra[s.d, specIndex, 2] * 1e6
            #cell_T = s.spectra[s.d, specIndex, 3] * 1e1
            irradiance = s.P[s.d, specIndex]
            aoi = s.spectra[s.d, specIndex, 5] * 1e6
            aim = physicaliam(aoi)
            s.T = sandia_T(s.heat_eff() * aim * irradiance, wind, amb_T) + 273.15
            #print('aoi, amb_T, cell_T, irradiance, wind, s.T',aoi, amb_T, cell_T, irradiance, wind, s.T, end="\r")
            #print('gaps '+str(s.gaps)+' s.T[0] '+str(s.T[0]), end="\r")
        else:
            # To a first approximation, cell T is a linear function of irradiance.
            s.T = s.Tmin + s.deltaT[s.d] * s.P[s.d, specIndex] / 1000

        # From top to bottom: get external photocurrent in each junction
        for i in range(s.junctions-1, -1, -1):
            # IntSpec = Integrated current from UV to given Energy gap
            IntSpec = s.intSpec(s.gaps[i] + Varshni(s.T), specIndex)
            # Get external I per junction
            Ijx0 = IntSpec-upperIntSpec
            upperIntSpec = IntSpec
            if i != 0:
                # Subcell thickness cannot be infinite,
                # 3 micron GaAs has 2% transmission
                Ijx[i] = (1-s.transmission)*Ijx0+IfromTop
                IfromTop = s.transmission*Ijx0
            else:
                # bottom junction does not transmit (back mirror)
                Ijx[i] = Ijx0+IfromTop
        return Ijx

    def thin(s):
        """ Calculate transmission factors for each subcell to
        maximize current under spectrum given in .thinSpec
        """
        # Only of interest in a few special cases with constrained band gaps.
        # THIS FUNCTION IS NOT FULLY TESTED AND SELDOM USED
        # Top cell thinning:
        # - From top to bottom junction
        #       - If next current is lower:
        #             - Find average with next junction
        #             - If average I is larger than next junction I:
        #                   - extend average and repeat this step
        # Most general case considered is two mechanically stacked devices.
        # Each device has 2 terminals. Stack has 4 terminals.

        Ijx = s.getIjx(s.thinSpec)  # get external photocurrent
        initialIjx = np.copy(Ijx)
        bottomJts = s.junctions-s.topJunctions  # top stack bottom junction
        # stack is the top 2 terminal multijunction in the mechanical stack
        stack = [[s.junctions-1, bottomJts]]
        if bottomJts > 0:  # If there are additional junctions:
            # Add the bottom 2 terminal multijunction in the mechanical stack
            stack.append([bottomJts-1, 0])
        for seriesDevice in stack:  # First do top device in mechanical stack
            topJ, bottomJ = seriesDevice
            Ijxmin = 0
            # While min I keeps going up
            while int(Ijxmin * 100) < int(Ijx[bottomJ:topJ + 1].min() * 100):
                Ijxmin = Ijx[bottomJ:topJ + 1].min()
                donorSubcell = topJ
                while donorSubcell > bottomJ:  # Spread I from top to bottom
                    # for aceptorSubcell in range(donorSubcell-1, bottomJ-1, -1 ):
                    aceptorSubcell = donorSubcell - 1
                    # If next current is lower:
                    if Ijx[donorSubcell] > Ijx[aceptorSubcell]:
                        # Average current with lower junction to reduce excess
                        mean = np.mean(Ijx[aceptorSubcell:donorSubcell + 1])
                        previousMean = mean + 1
                        # Try to decrease excess in mean by including lower subcells
                        while mean < previousMean:
                            previousMean = mean
                            aceptorSubcell -= 1  # include lower subcell in mean
                            if aceptorSubcell < bottomJ:  # sanity check
                                aceptorSubcell = bottomJ
                            else:
                                mean = np.mean(Ijx[aceptorSubcell: donorSubcell + 1])
                        # Backtrack if last attempt was failure
                        if previousMean < mean:
                            mean = previousMean
                            aceptorSubcell += 1
                        # Change currents
                        Ijx[aceptorSubcell: donorSubcell + 1] = mean
                        # Jump over subcells that have been averaged
                        donorSubcell = aceptorSubcell
                    donorSubcell -= 1 # Go down

            if bottomJts > 0:  # if there is a lower series_connected device in stack
                # The excess current from the bottom junction of the top series_connected, goes to lower series_connected
                exIb = Ijx[bottomJts] - Ijx[bottomJts:].min()
                Ijx[bottomJts] -= exIb
                Ijx[bottomJts-1] += exIb

        s.thinTrans = Ijx / initialIjx

    def timePerSpectra(s, numBins, binIndex):
        """ Returns fraction of yearly daytime represented by each spectra """
        if numBins < binMax:
            return s.timePerCluster[s.d, getSpectrumIndex(numBins, binIndex)]
        else:
            return 1 / s.numSpectra[numBins]

    def series(s, topJ, bottomJ, numBins, binIndex, concentration=1):
        """ Get power from 2 terminal monolythic multijunction device.
        Series connected subcells with indexes topJ to bottomJ.
        topJ = bottomJ is single junction.
        """
        # 1 - Get external photocurrent in each junction
        # 2 - Get current at the maximum power point from min external photocurrent
        # 3 - Recalculate maximum power point including radiative coupling
        # 4 - This changes radiative coupling, repeat 3 until self consistency
        # 5 - Calculate power out

        # Do "tandems.show_assumptions()" to see models for EQE, AOD, and PW

        if (topJ < 0):
            return np.array([0]), np.array([0])
        Ijx = concentration * s.getIjx(getSpectrumIndex(numBins, binIndex)) * s.thinTrans
        # thinTrans is set by thin()
        # s.T is set by s.getIjx
        # For a 1mm2 cell at 1000 suns bonded to copper substrate T = 70
        # See CPV Handbook, ed. by: I. Rey-Stolle, C. Algora
        kT = con.k*s.T
        k0 = 2 * np.pi * q * kT**3 / (con.h * hc**2)
        Irc0 = s.Irc  # radiative coupling from upper stack

        if Ijx.sum() == 0:
            return np.array([0]), np.array([0])

        # if Ijx.min() <= 0:
            # pdb.set_trace()
        # Initial guess for current at the maximum power point
        Imax = Ijx[bottomJ:topJ + 1].min()
        Imaxs = [0, Imax]
        # Loop to self consistently refine max power point
        Ijmin = 0
        I_sample = np.array([0])
        V = 0
        while (((Imax-Imaxs[-2]) / Imax)**2) > 1e-7:
            V = 0
            s.Irc = Irc0  # Radiative coupling from upper stack
            Ij = np.copy(Ijx)  # Current absorbed in each junction
            # From top to bottom: get photocurrent in each junction including radiative coupling
            for i in range(topJ, bottomJ-1, -1):
                Ij[i] += s.Irc  # Include emitted light from upper junction
                if (Ij[i] > Imax):  # If there is excess current in this junction:
                    s.Irc = s.beta*s.ERE*(Ij[i]-Imax)  # radiative coupling
                else:
                    s.Irc = 0
            Ijmin = Ij[bottomJ:topJ+1].min()  # Min current in series connected stack
            I_sample = Ijmin*np.arange(0.8, 1, 0.0001)  # IV curve sampling
            # From top to bottom: Sample IV curve, get I0
            for i in range(topJ, bottomJ-1, -1):
                # The bottom junction of each series connected stack has photon recycling due to partial back reflection of luminescence
                if (i == bottomJ) and not s.opticallyCoupledStacks:
                    # Electroluminescence lost at the back of the bottom junction normalized to ERE
                    backLoss = s.mirrorLoss
                else:
                    # Electroluminescence transfered to the next lower junction normalized to ERE
                    backLoss = s.beta
                g0 = s.gaps[i] * q / kT
                # Recombination current at V = 0 in A / m2 s
                I0 = (1 + backLoss) * k0 * np.exp(-1 * g0) * ((g0 + 1)**2 + 1) / s.ERE
                # add voltage of series connected cells
                V += (kT / q) * np.log((Ij[i] - I_sample)/I0 + 1)
            V -= s.R*I_sample # V drop due to series resistance
            Imax = I_sample[np.argmax(I_sample * V)]  # I at max power point
            # The following can be used to monitor IV curve parameters
            # print(Imax, V[np.argmax(I_sample * V)], I_sample[0], I_sample[-1], V[0], V[-1], (kT / q) * np.log((Ij[i]) / I0 + 1))
            Imaxs.append(Imax)
        if len(Imaxs) > 10:
            print('self consistency is slowing convergence while finding the maximum power point.')
            print('ERE or beta might be too high.')
            print('Current at the maximum power point is converging as:', Imaxs)
            pdb.set_trace()
        # Yearly daytime average photocurrent per unit cell area
        s.Itotal += Ijmin * s.timePerSpectra(numBins, binIndex)
        # Yearly daytime average power per unit module area
        s.Pout += (I_sample * V).max() * s.timePerSpectra(numBins, binIndex) / concentration
        return I_sample, V

    # A stack is composed of one monolythic two terminal device or two of them mechanically stacked
    def stack(s, numBins, binIndex):
        """ Use a single spectrum to get power from 2, 3, or 4 terminal tandem. If topJunctions = junctions the result is for 2 terminal tandem. """
        # Most general case considered is two mechanically stacked devices. Each device has 2 terminals. Stack has 4 terminals.
        s.Irc = 0  # For top cell there is no radiative coupling from upper cell
        pout = s.Pout
        # For top stack, calculate power out for each spectra and add to Pout. topJunctions is number of juntions in top stack
        I, V = s.series(s.junctions-1, s.junctions-s.topJunctions, numBins, binIndex, s.concentration)
        if not s.opticallyCoupledStacks:
            s.Irc = 0
        # For bottom stack, calculate power out for each spectra and add to Pout.
        I2, V2 = s.series(s.junctions-s.topJunctions-1, 0, numBins, binIndex, s.concentration)
        s.Pbins[binIndex] = s.Pout - pout
        return I + I2, V2

    def useBins(s, bins, results):
        """Use spectral bins and s.gaps to calculate yearly energy yield and eff.
        Optionally discard bad band gap combinations (results).
        """
        for numBins in bins:
            Pin = 0
            s.Itotal = 0
            s.Pout = 0
            # numSpectra = [1, 1, 2, 3, ...
            for binIndex in range(s.numSpectra[numBins]-1, -1, -1):
                # Get power in/out for each spectra and add to Pin and Pout
                s.stack(numBins, binIndex)
                # Yearly averaged daytime power in from the sun
                Pin += s.P[s.d, getSpectrumIndex(numBins, binIndex)] * s.timePerSpectra(numBins, binIndex)
            eff = s.Pout / Pin
            s.effs[results, numBins] = eff
            s.Is[results, numBins] = s.Itotal
        s.rgaps[results,:] = s.gaps
        s.auxEffs[results, :] = np.zeros(s.junctions)+s.effs[results, numBins]
        s.auxIs[results, :] = np.zeros(s.junctions)+s.Itotal
        # Will overwrite this result if it is not good enough
        if eff < (s.effs[:, numBins].max() - s.effMin):
            return results
        return results + 1

    def mask(s, mask):
        """ Discard or sort results according to mask
        """
        s.rgaps = s.rgaps[mask, :]
        s.auxIs = s.auxIs[mask, :]
        s.auxEffs = s.auxEffs[mask, :] # [:, 0]
        s.effs = s.effs[mask, :]
        s.Is = s.Is[mask, :]
        # As a side effect of this masking, arrays are flattened
        s.rgaps = np.reshape(s.rgaps, (-1, s.junctions))  # Restore proper array shapes after masking
        s.auxIs = np.reshape(s.auxIs, (-1, s.junctions))
        s.auxEffs = np.reshape(s.auxEffs, (-1, s.junctions))
        s.effs = np.reshape(s.effs, (-1, binMax + 1))
        s.Is = np.reshape(s.Is, (-1, binMax + 1))

    def findGaps(s):
        """ Calculate efficiencies for random band gap combinations. """
        startTime = time.time()
        ncells = 0  # Number of calculated gap combinations
        results = 0
        effmax = 0

        # These arrays are used to speed up calculations by limiting the energy gap search space.
        # Initial guess at where the eff maxima are.
        # The search space is expanded as needed if high eff are found at the edges of the search space.
        Emax = []  # Max energy for each gap
        Emin = []  # Min energy for each gap
        Emax.append([1.6])  # 1 junction
        Emin.append([0.9])
        Emax.append([1.40, 2.00])  # 2  junctions
        Emin.append([0.85, 1.40])
        Emax.append([1.20, 1.67, 2.00])  # 3  junctions
        Emin.append([0.65, 1.15, 1.60])
        Emax.append([1.10, 1.51, 1.90, 2.10])  # 4  junctions
        Emin.append([0.50, 0.94, 1.25, 1.80])
        Emax.append([0.95, 1.15, 1.55, 2.00, 2.25])  # 5  junctions
        Emin.append([0.50, 0.83, 1.15, 1.50, 1.90])
        Emax.append([0.95, 1.15, 1.45, 1.85, 2.10, 2.30])  # 6  junctions
        Emin.append([0.50, 0.78, 1.05, 1.40, 1.70, 2.05])

        # These arrays are used to discard band gap combinations that are too unevenly spaced in energy
        # The search space is expanded as needed if high eff are found at the edges of the search space.
        maxDif = []
        minDif = []
        maxDif.append([1.00])  # 2 junctions, max E difference between junctions
        minDif.append([0.70])  # 2 junctions, min E difference between junctions
        maxDif.append([0.65, 0.55])
        minDif.append([0.50, 0.50])  # 3 junctions
        maxDif.append([0.60, 0.55, 0.65])
        minDif.append([0.45, 0.45, 0.55])  # 4 junctions
        maxDif.append([0.50, 0.60, 0.45, 0.55])
        minDif.append([0.30, 0.40, 0.40, 0.45])  # 5 junctions
        maxDif.append([0.50, 0.50, 0.40, 0.40, 0.50])
        minDif.append([0.28, 0.28, 0.32, 0.32, 0.42])  # 6 junctions

        Emin_ = np.array(Emin[s.junctions-1])  # Initial guess at where the eff maxima are.
        Emax_ = np.array(Emax[s.junctions-1])  # The search space is expanded as needed if high eff are found at the edges of the search space.
        minDif_ = np.array(minDif[s.junctions-2])
        maxDif_ = np.array(maxDif[s.junctions-2])
        # Remove comment on np.random.seed if you want to reuse the same sequence of random numbers (e.g.: compare results after changing one parameter)
        # np.random.seed(07022015)
        s.rgaps = np.zeros((s.cells, s.junctions))  # Gaps
        s.auxIs = np.zeros((s.cells, s.junctions))  # Aux array for plotting only
        s.auxEffs = np.zeros((s.cells, s.junctions))  # Aux array for plotting only
        s.Is = np.zeros((s.cells, binMax + 1))  # Currents as a function of the number of spectral bins, 0 is standard spectrum, binMax is full spectral dataset
        s.effs = np.zeros((s.cells, binMax + 1))  # Efficiencies as a function of the number of spectral bins, 0 is standard spectrum, binMax is full spectral dataset
        fixedGaps = np.copy(s.gaps)  # Copy input gaps to remember which ones are fixed, if gap==0 make it random

        while (results<s.cells):  # Loop to randomly sample a large number of gap combinations
            s.gaps = np.zeros(s.junctions)
            lastgap = 0
            i = 0
            while i<s.junctions:     # From bottom to top: define random gaps
                if i>0:
                    Emini = max(Emin_[i], lastgap + minDif_[i-1])  # Avoid gap combinations that are too unevenly spaced
                    Emaxi = min(Emax_[i], lastgap + maxDif_[i-1])
                else:
                    Emini = Emin_[i]
                    Emaxi = Emax_[i]
                Erange = Emaxi - Emini
                if fixedGaps[i] == 0:
                    s.gaps[i] = Emini + Erange * np.random.rand()  #define random gaps
                else:
                    s.gaps[i] = fixedGaps[i]  # Any gaps with an initial input value other than 0 are kept fixed, example: tandems.effs(gaps=[0.7, 0, 0, 1.424, 0, 2.1])
                if i>0 and fixedGaps.sum():  # If there are fixed gaps check gaps are not too unevenly spaced
                    if not (0.1 < s.gaps[i]-lastgap < 1):  # gap difference is not in range
                        lastgap = 0  # Discard gaps and restart if gaps are too unevenly spaced
                        i = 0
                    else:
                        lastgap = s.gaps[i]
                        i += 1
                else:
                    lastgap = s.gaps[i]
                    i += 1

            if s.thinning:
                s.thin()

            s.useBins([1], results)
            eff = s.effs[results, 1]  # get rough estimate of efficiency

            if (eff > effmax - s.effMin - 0.01):  # If gap combination is good, do more work on it
                for gi, gap in enumerate(s.gaps):  # Expand edges of search space if any of the found gaps are near an edge
                    if gap < Emin_[gi] + 0.01:  # Expand energy range allowed for each gap as needed
                        if Emin_[gi] > 0.5: # Only energies between 0.5 and 2.5 eV are included
                            Emin_[gi] -= 0.01
                    if gap > Emax_[gi] - 0.01:
                        if Emax_[gi] < 2.5:
                            Emax_[gi] += 0.01
                    if gi > 0:
                        if gap - s.gaps[gi-1] + 0.01 > maxDif_[gi-1]:  # Allow for more widely spaced gaps if needed
                            maxDif_[gi-1] += 0.01
                        if gap - s.gaps[gi-1] - 0.01 < minDif_[gi-1]:  # Allow for more closely spaced gaps if needed
                            minDif_[gi-1] -= 0.01
                if (eff>effmax):
                    effmax = eff
                results = s.useBins(s.bins, results)  # Calculate efficiency for s.gaps, store result if gaps are good
                #if results % 8 == 0:  # Show calculation progress from time to time
                    #print('Tried '+str(ncells)+', got '+str(results)+' # ', end="\r\r")
            ncells += 1

        threshold = s.effs[:, s.bins[-1]].max() - s.effMin
        mask = s.effs[:, s.bins[-1]] > threshold  # Discard results below the efficiency threshold set by effMin
        s.mask(mask)

        tiempo = int(time.time()-startTime)
        res = np.size(s.rgaps)/s.junctions
        print('Calculated ', ncells, ' and saved ', res, ' gap combinations in ', tiempo, ' s :', res/(tiempo+1), ' results/s')
        s.results()

    def kWh(s, efficiency):
        """Converts efficiency values to yearly energy yield in kWh/m2"""
        return 365.25*24 * s.P[s.d, 1] * efficiency * (1 - s.cloudCover) * s.daytimeFraction / 1000

    def results(s):
        """ After findGaps() or recalculate(), or load(), this function shows the main results """
        print('---' + s.name + '---')
        print('Maximum efficiency:', s.auxEffs.max())
        print('Maximum yearly energy yield:', s.kWh(s.auxEffs.max()))
        print('P sun',365.25*24 * s.P[s.d, 1] * (1 - s.cloudCover) * s.daytimeFraction / 1000, 'cloudCover', s.cloudCover, 'daytimeFraction', s.daytimeFraction)
        imax = np.argmax(s.auxEffs[:, 0])
        print('Optimal gaps:', s.rgaps[imax])
        print('Isc for optimal gaps (A/m2):', s.auxIs[imax, 0])
        print('Isc range (A/m2):', s.auxIs.min(), '-', s.auxIs.max())

    def plot(s, dotSize=1, show=True):
        """ Saves efficiency plots to PNG files """
        res = np.size(s.rgaps)/s.junctions
        Is_ = np.copy(s.auxIs)
        Is_ = (Is_-Is_.min())/(Is_.max()-Is_.min())
        if s.convergence:
            ec = 100*(s.effs[:, 1:-1][:, np.delete(s.bins, -2) - 1] - s.effs[:, -1][:, None])
            print(s.bins)
            ecm = np.median(ec, axis=0)
            emax = np.ndarray.max(ec, axis=0)
            emin = np.ndarray.min(ec, axis=0)
            eq1 = np.percentile(ec, 25, axis=0)
            eq3 = np.percentile(ec, 75, axis=0)
            print('Median and maximum error',ecm[-1], emax[-1])
            plt.figure()
            plt.xlabel('Number of spectra')
            plt.ylabel('Yearly efficiency error (%)')
            plt.xlim(0.3, binMax - 0.3)
            plt.ylim(0, 1.05*(100 * (s.effs[: , 1] - s.effs[: , -1])).max())
            #plt.ylim(0.02, 4)
            plt.tick_params(axis='y', right='on')
            ax = plt.gca()
            #ax.set_yscale('log')
            ax.xaxis.set_major_locator(plt.MultipleLocator(2))
            #plt.plot([-1, 32], [0, 0], linestyle=':', color='black')
            for i in range(0, int(res)):
                plt.scatter(np.delete(s.bins, -2), ec[i, :], color = np.array([0,0,0,1]), s=100, marker='_', linewidth=0.3)
            plt.savefig(s.name + s.specsFile.replace('.npy', '').replace('.','-') + ' convergence ' + s.filenameSuffix, dpi=300, bbox_inches="tight")
            plt.savefig(s.name + s.specsFile.replace('.npy', '').replace('.','-') + ' convergence ' + s.filenameSuffix + '.svg', bbox_inches="tight")
            if show:
                plt.show()

            if False: # Set to True to plot quartiles
                plt.figure()
                plt.xlabel('Number of spectra')
                plt.ylabel('Yearly efficiency overestimate (%)')
                plt.xlim(1, binMax - 1)
                plt.ylim(0.02, 4)
                ax = plt.gca()
                #plt.grid(which='major')
                ax.set_yscale('log')
                ax.xaxis.set_major_locator(plt.MultipleLocator(2))
                plt.tick_params(axis='y', right='on')
                plt.plot(s.bins, ecm, color = '#0077BB')
                plt.fill_between(s.bins, eq1 , eq3, facecolor=np.array([0, 0.6, 0.8, 0.32]))
                plt.fill_between(s.bins, emin , emax, facecolor=np.array([0, 0.6, 0.8, 0.32]))
                plt.savefig(s.name + s.specsFile.replace('.npy', '').replace('.','-') +
                            ' convergence_ ' + s.filenameSuffix, dpi=300, bbox_inches="tight")
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
        plt.scatter(s.rgaps, 100 * s.auxEffs, c=Is_, s = dotSize * 100000 / len(s.effs[:, s.bins[-1]]), edgecolor='none', cmap=LGBT)
        axb = plt.gca().twinx()
        axb.set_ylim(s.kWh(ymin),s.kWh(ymax))
        axb.set_ylabel('Yield $\mathregular{(kWh \ m^{-2} \ year^{-1})}$')
        plt.savefig(s.name + s.specsFile.replace('.npy', '').replace('.','-') +
                    ' ' + s.filenameSuffix, dpi=300, bbox_inches="tight")

        plt.figure()
        plt.ylim(100*ymin, 100*ymax)
        plt.minorticks_on()
        plt.tick_params(direction='out', which='minor')
        plt.tick_params(direction='inout', pad=6)
        plt.ylabel('Yearly averaged efficiency (%)')
        plt.xlabel('Daytime average short circuit current $\mathregular{(mA \ cm^{-2})}$')
        Is_ = np.copy(s.Is[:, s.bins[-1]])
        Is_ = (Is_-Is_.min())/(Is_.max()-Is_.min())
        plt.scatter(s.Is[:, s.bins[-1]]/10, 100 * s.effs[:, s.bins[-1]], c=Is_,
                    s = dotSize * 100000 / len(s.effs[:, s.bins[-1]]), edgecolor='none', cmap=LGBT)
        axb = plt.gca().twinx()
        axb.set_ylim(s.kWh(ymin),s.kWh(ymax))
        axb.set_ylabel('Yield $\mathregular{(kWh \ m^{-2} \ year^{-1})}$')
        plt.savefig(s.name + s.specsFile.replace('.npy', '').replace('.','-') +
                    ' I ' + s.filenameSuffix, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        if False:  # Set to True to plot histograms
            plt.figure()
            plt.xlim(100*(s.auxEffs.max()-s.effMin), 100*s.auxEffs.max())
            plt.xlabel('Yearly averaged efficiency (%)')
            plt.ylabel('Count')
            plt.hist(100*s.effs[:, s.bins[-1]], bins=30)
            plt.savefig(s.name + s.specsFile.replace('.npy', '').replace('.', '-') +
                        ' Hist ' + s.filenameSuffix, dpi=300, bbox_inches="tight")
        if show:
            plt.show()

    def save(s, saveSpectra=False):
        """ Saves data for later reuse/replotting. Path and file name set in eff.name, some parameters and autoSuffix are appended to filename """
        s2 = s
        if not saveSpectra:
            s2 = deepcopy(s)
            s2.spectra = 0
            s2.Iscs = 0
        with open(s.name + s.specsFile.replace('.npy', '')+' '+s.filenameSuffix, "wb") as f:
            pickle.dump(s2, f, pickle.HIGHEST_PROTOCOL)

    def purge(s, spanFraction=500, effThreshold=0.001):
        """ Delete suboptimal devices. Only devices with efficiency within 0.1% (effThreshold) of the maximum at each current are retained
        This is useful before recalculate and compare.
        """
        newOrder = np.argsort(s.Is[:, s.bins[-1]])  # Sort all data in order of increasing current
        s.mask(newOrder)

        span = s.Is.shape[0] / spanFraction  # Number of data points when finding the maximum efficiency at each current

        maxEffs = np.copy(s.effs[:, s.bins[-1]])  # Create new array for storing the efficiency threshold at each current
        # Find efficiency threshold at each current given by percentile of the
        # population given by +/- span datapoints around each current
        # adjust span to be a small fraction of the total number of data points
        for i, eff in enumerate(s.effs[:, s.bins[-1]]):
            mini = max(0, i - int(span / 2))
            maxi = min(s.effs[:, s.bins[-1]].shape[0], i + int(span / 2))
            maxEffs[i] = np.max(s.effs[mini:maxi, s.bins[-1]])
        mask = s.effs[:, s.bins[-1]] > maxEffs - effThreshold
        s.mask(mask)

    def recalculate(s):
        """ Recalculate efficiencies with new set of input parameters using previously found gaps. Call __init__() to change parameters before recalculate() """
        for gi in range(0, s.rgaps.shape[0]):
            s.gaps = s.rgaps[gi]
            s.useBins(s.bins, gi)  # Calculate efficiency for s.gaps
        s.results()

    def compare(s, s0, dotSize=1, show=True, r1=(0.8, 0.1, 0, 0.8), r2=(1, 0.1, 0, 0.4), b1=(0, 0.1, 1, 0.8), b2=(0, 0.1, 1, 0.4)):
        """ Plots relative differences of effiency for two results sets based on the same set of optimal band gap combinations. """
        print('I min, I max : ', s.auxIs.min(), s.auxIs.max())
        print('eff min, eff max : ', s.auxEffs.min(), s.auxEffs.max())
        Is_ = np.copy(s0.auxIs)
        Is_ = (Is_-Is_.min())/(Is_.max()-Is_.min())
        if s.convergence:
            plt.figure()
            plt.xlabel('Number of spectra')
            plt.ylabel('Yearly efficiency overestimate (%)')
            plt.xlim(1, binMax - 1)
            ejes = plt.gca()
            ejes.xaxis.set_major_locator(plt.MultipleLocator(2))
            plt.tick_params(axis='y', right='on')
            ec = 100*(s.effs[:, 1:-1] - s.effs[:, -1][:, None]) # IN CASE OF ERROR: ec = ec[:, np.delete(s.bins, -2)] might be needed after this
            ecm = np.mean(ec, axis=0)
            ece = np.sqrt(np.sum((ec - ecm)**2, axis=0) / (ec.shape[0] - 1))
            plt.ylim(0, 1.05*(ecm + ece)[1])
            plt.scatter(s.bins, ecm + ece, marker='v', s=80, c=b2)
            plt.scatter(s.bins, ecm, marker='_', s=80, c=b1)  # color = LGBT(auxIs[i*s.junctions])
            ec = 100*(s0.effs[: , 1:-1] - s0.effs[: , -1][:, None])
            ecm = np.mean(ec, axis=0)
            ece = np.sqrt (np.sum((ec - ecm)**2, axis = 0) / (ec.shape[0] - 1))
            plt.scatter(s.bins, ecm + ece, marker='v', s=80, c=r2)
            plt.scatter(s.bins, ecm, marker='_', s=80, c=r1)  # color = LGBT(auxIs[i*s.junctions])
            plt.savefig(s.name + s.specsFile.replace('.npy', '').replace('.', '-') +
                        ' convergence compare ' + s.filenameSuffix, dpi=300, bbox_inches="tight")
            if show:
                plt.show()
        else:
            plt.figure()
            plt.minorticks_on()
            plt.tick_params(direction='out', which='minor')
            plt.tick_params(direction='inout', pad=6)
            plt.ylabel('Energy yield change (%)')
            percentChange = 100 * (s.kWh(s.auxEffs) - s0.kWh(s0.auxEffs)) / s0.kWh(s0.auxEffs)
            plt.xlabel('Daytime average short circuit current $\mathregular{(mA \ cm^{-2})}$')
            plt.scatter(s0.auxIs/10, percentChange, c=Is_, s=dotSize*20000/len(s.effs[:, s.bins[-1]]), edgecolor='none', cmap=LGBT)
            plt.savefig(s.name + '-' + s0.name.replace('/', '').replace('.', '') + ' ' +
                        s.specsFile.replace('.npy', '').replace('/', '').replace('.', '_') +
                        '-' + s0.specsFile.replace('.npy', '').replace('/', '').replace('.', '_') +
                        ' ' + str(int(s0.junctions)) + ' ' + str(int(s0.topJunctions)) +
                        ' ' + str(int(s0.concentration)) + ' ' + s.autoSuffix,
                        dpi=300, bbox_inches="tight")
            if show:
                plt.show()
# ---- End of Class effs ----


def load(fnames):
    """ Load previously saved effs objects.
    Args:
        fnames (str): File name,
        Dptah (str, optional): Folder to look for the files

    tandems.load('/path/and_file_name_pattern*here')
    A file name pattern with wildcards (*) can be used to
    load a number of files and join them in a single object.
    This is useful for paralelization.
    All files need to share the same set of input parameters because this function
    does not check if it makes sense to join the output files.
    """
    s = False
    arrayOfFiles = glob(fnames)
    if arrayOfFiles:
        for fname in arrayOfFiles:
            with open(fname, "rb") as f:
                if not s:
                    s = pickle.load(f)
                else:
                    s2 = pickle.load(f)
                    s.rgaps = np.append(s.rgaps, s2.rgaps, axis=0)
                    s.auxIs = np.append(s.auxIs, s2.auxIs, axis=0)
                    s.auxEffs = np.append(s.auxEffs, s2.auxEffs, axis=0)
                    s.Is = np.append(s.Is, s2.Is, axis=0)
                    s.effs = np.append(s.effs, s2.effs, axis=0)
        s.cells = s.rgaps.shape[0]
    else:
        print('Files not found')
    return s


def multiFind(cores=2, saveAs='someParallelData',
              parameterStr='cells=1000, junctions=4'):
    """Hack to use as many CPU cores as desired in the search for optimal band gaps.
    Uses the tandems.effs object and its findGaps method.
    Total number of results will be cores * cells.
    Calls bash (so do not expect this function to work under Windows)
    Creates many instances of this python module running in parallel.
    """
    os.system('for i in `seq 1 ' + str(cores)
              + '`; do python -c "import tandems;s=tandems.effs(name='
              + chr(39)+saveAs+chr(39) + ','
              + parameterStr+');s.findGaps();s.save()" & done; wait')
    # Loads all calculation results and joins them in a single object
    s = load(saveAs+'*')
    s.results()
    s.plot()


def show_assumptions():  # Shows the used EQE model and the AOD and PW statistical distributions
    s0 = effs()
    plt.figure()
    plt.xlim(0.4, 4)
    plt.ylim(0, 1)
    plt.title('EQE model fitted to record device with 46% eff. \n DOI.: 10.1109/JPHOTOV.2015.2501729')
    plt.ylabel('External Quantum Efficiency')
    plt.xlabel('Photon energy (eV)')
    plt.plot(Energies, s0.EQE)
    plt.show()

    # AOD random distribution used here is fitted to histograms by Jaus and Gueymard based on Aeronet data
    lrnd = np.random.lognormal(-2.141, 1, size=1000000)
    lrnd[lrnd > 2] = 2  # For plotting purposes, AOD values are capped here at AOD=2, this is not done during calculations
    plt.figure()
    titel = 'AOD random distribution used here is fitted to \n histograms by Jaus and Gueymard based on Aeronet data, \n'
    titel += 'DOI: 10.1063/1.4753849 \n In Jaus and Gueymard, percentiles at 25% and 75% are 0.06, 0.23. \n'
    titel += 'Here: '+str(np.percentile(lrnd, 25))+' '+str(np.percentile(lrnd, 75))
    plt.title(titel)
    plt.xlim(0, 0.8)
    plt.xlabel('Aerosol Optical Depth at 500 nm')
    plt.hist(lrnd, bins=200)
    plt.show()

    # PW random distribution used here is fitted to histograms by Jaus and Gueymard based on Aeronet data
    lrnd = 0.39*np.random.chisquare(4.36, size=1000000)
    lrnd[lrnd > 7] = 7
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
                           speCount=365*24, NSRDBfile='', fname='spectra',
                           saveFullSpectra=False, loadFullSpectra=False,
                           method='clusters', dissolve=True,
                           killSmall=True, submethod='kmeans', numFeatures=0,
                           bins=np.arange(1,binMax,1),
                           glo_dir=[0,1]):
    """ Use file with full yearly set of spectra to generate a few characteristic proxy
    spectra using a clustering method (recommended for more than 10 proxy spectra)
    or a binning method (recommended for less than 6 proxy spectra).
    If no files are provided the spectra are generated usign a slightly modified
    version of SMARTS 2.9.5
    """

    # The SMARTS FOLDER/DIRECTORY SHOULD BE IN THE SAME PLACE AS tandems.py
    smartsDir = 'SMARTS/'

    def EPR(i, s1):
        """ Calculates EPR, integrates Power and stores spectra"""
        P[:, i] = integrate.trapz(s1[:, 7:], x=wavel[7:], axis=1) # the first few elements of the spectra are used to store temps and winds
        if (P[0, i] > 0):
            fullSpectra[:, i, :] = s1
            # Calculate EPR, Ivan Garcia's criteria for binning
            EPR650[:, i] = integrate.trapz(s1[:, 7:490], x=wavel[7:490], axis=1)
            EPR650[:, i] = (P[:, i] - EPR650[:, i]) / EPR650[:, i]
            i += 1  # Spectrum index is incremented if spectrum is not zero
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

    fullSpectra = np.zeros((2, speCount, len(wavel)))  # Array to hold generated spectra, first index is 0=global or 1=direct
    EPR650 = np.zeros((2, speCount))  # Ivan Garcia's criteria for binning, store EPR value for each spectrum
    P = np.zeros((2, speCount))  # Power for each spectrum

    # Optionally Load specified National Solar Resource DataBase spectra
    if NSRDBfile != '':
        # https://maps.nrel.gov
        Ng = np.loadtxt(Dpath + NSRDBfile, skiprows=3, delimiter=',',
                        usecols=tuple(range(159, 310)))  # Global Horizontal
        Nd = np.loadtxt(Dpath + NSRDBfile, skiprows=3, delimiter=',',
                        usecols=tuple(range(8, 159)))  # Direct spectra
        Tw = np.loadtxt(Dpath + NSRDBfile, skiprows=3, delimiter=',',
                        usecols=(6, 7))  # Ambient T, wind
        attempts = Nd.shape[0]
        if fname == 'spectra':
            fname = NSRDBfile.replace('.csv', '')
        s1 = np.zeros((2, len(wavel)))
        speCount = 0
        for i in range(0, len(Nd)):
            s1[0, :] = np.interp(wavel, np.arange(300, 1810, 10), Ng[i, :],
                                 left=0, right=0)  # Global Horizontal
            s1[1, :] = np.interp(wavel, np.arange(300, 1810, 10), Nd[i, :],
                                 left=0, right=0)  # Direct spectra
            speCount = EPR(speCount, s1)  # Calculates EPR, integrates Power and stores spectra
            print(speCount, 'spectra out of', attempts, 'points in time         ', end="\r")
        fullSpectra = fullSpectra[:, :speCount, :]  # Array to hold the whole set of spectra
        EPR650 = EPR650[:, :speCount]  # Ivan Garcia's criteria for binning, store EPR value for each spectrum
        P = P[:, :speCount]  # Power for each spectrum
        np.save(fname+'.Tw', Tw)

    elif loadFullSpectra:  # To reuse a full set of spectra saved earlier
        fullSpectra_loaded = load_np(fname+'.full.npz')
        EPR650 = np.zeros((2, fullSpectra_loaded.shape[1]))  # Ivan Garcia's criteria for binning, store EPR value for each spectrum
        P = np.zeros((2, fullSpectra_loaded.shape[1]))  # Power for each spectrum
        attempts = fullSpectra_loaded [-1, -1, -1]

        speCount = 0
        for specIndex in range(0, fullSpectra_loaded.shape[1]):
            speCount = EPR(speCount, fullSpectra_loaded[:, specIndex, :])  # Calculates EPR, integrates Power and stores spectra
        fullSpectra = fullSpectra[:, :speCount, :]  # Array to hold the whole set of spectra
        EPR650 = EPR650[:, :speCount]  # Ivan Garcia's criteria for binning, store EPR value for each spectrum
        P = P[:, :speCount]  # Power for each spectrum
        fullSpectra[-1, -1, -1] = 0

    else:  # Generate SMARTS spectra if no file with spectra is provided
        print('Path is', os.path.abspath(os.curdir))
        longitude2 = longitude
        AOD2 = AOD
        PW2 = PW
        with open(Dpath + 'smarts295.in_.txt', "r") as fil:  # Load template for SMARTS input file
            t0 = fil.readlines()
        t0 = ''.join(t0)
        attempts = 0
        t0 = t0.replace('38 -999 -999', tracking)
        for specIndex in range(0, speCount):
            tama = 0
            while tama == 0:  # Generate random time, location, AOD, PW

                if longitude == 'random':
                    longitude2 = 360*np.random.rand()-180
                # To plot AOD and PW distributions do "tandems.show_assumptions()"
                if AOD == 'random':
                    AOD2 = min(5.4, np.random.lognormal(-2.141, 1))
                if PW == 'random':
                    PW2 = min(12, 0.39*np.random.chisquare(4.36))

                # Substitute AOD and PW into template SMARTS input file
                t1 = t0.replace('AOD#', str(AOD2))
                t1 = t1.replace('PW#', str(PW2))
                # from 7/feb/2018 to 7/feb/2019
                t2 = datetime.fromtimestamp(3155673600.0*np.random.rand()+1517958000.0).strftime("%Y %m %d %H")+' '
                t2 += '%.2f' % ((latMax-latMin)*np.arccos(2*np.random.rand()-1)/np.pi+latMin)+' '
                t2 += '%.2f' % (longitude2)+' 0\t\t\t!Card 17a Y M D H Lat Lon Zone\n\n'
                with open(smartsDir + 'smarts295.inp.txt', 'w+') as fil:
                    fil.write(t1+t2)
                try:
                    os.remove(smartsDir + 'smarts295.ext.txt')
                except OSError:
                    pass
                try:
                    os.remove(smartsDir + 'smarts295.out.txt')
                except OSError:
                    pass
                attempts += 1
                # execute SMARTS
                os.chdir(smartsDir)
                sub.check_call('./smartsAM4', shell=True)
                os.chdir('../')
                # check output file
                try:  # load output file if it exists
                    tama = os.stat(smartsDir + 'smarts295.ext.txt').st_size
                    if tama>0:
                        s1 = np.loadtxt(smartsDir + 'smarts295.ext.txt', delimiter=' ', usecols=(0, 1, 2), unpack=True, skiprows=1)
                        # Global Tilted = 1, Dir. Normal= 2, Diff. Horizontal=3
                except:
                    tama = 0
            EPR(specIndex, s1[1:, :])  # Calculates EPR, integrates Power and stores spectra
        print('Finished generating spectra. Called Smarts '+str(attempts)+' times and got '+str(speCount)+' spectra')

    if saveFullSpectra:
        fullSpectra[-1, -1, -1] = attempts
        np.savez_compressed(fname+'.full', fullSpectra)

    totalPower = np.zeros(2)
    RMS = np.zeros((2, binMax - 1))
    binlimits = []
    spectra = np.zeros((2, specMax, len(wavel)))  # Prepare data structure for saving to file
    timePerCluster = np.zeros((2, specMax))  # Each spectrum is representative of this fraction of yearly daytime

    if method == 'clusters':  # Use machine learning clustering methods

        for d in glo_dir:  # d = 1 for direct spectra
            specFeat = np.copy(fullSpectra)
            if numFeatures != 0:
                # Set numFeatures = 50 to speed up the clustering or 0 to skip this optional step
                # Search for features in the spectra that show the same trends as a function of time
                # Connectivity matrix only allows adjacent points in each feature
                conn = np.zeros((len(wavel), len(wavel)), dtype=int)

                for i in range(1, len(wavel)-1):
                    conn[i, i] = 1
                    conn[i, i+1] = 1
                    conn[i, i-1] = 1
                conn[0, 0] = 1
                conn[0, 1] = 1
                conn[len(wavel)-1, len(wavel)-1] = 1
                conn[len(wavel)-1, len(wavel)-2] = 1

                specFeat = np.zeros((2, speCount, numFeatures))
                features = FeatureAgglomeration(n_clusters=numFeatures,
                                                connectivity=conn
                                                ).fit(fullSpectra[d, :, :])

                for fea in range(numFeatures):
                    # Boolean array labeling points belonging in each feature
                    mask = features.labels_ == fea
                    # Integrated power in each feature
                    specFeat[d, :, fea] = fullSpectra[d, :, mask].sum(axis=0)

            specFeat[d, :, :] = normalize(specFeat[d, :, :])

            # Now merge simillar spectra

            for clustersCount in bins:  # bins is number of clusters
                specIndex = getSpectrumIndex(clustersCount, 0)
                # use extra clusters and then disolve smaller clusters
                extraClusters = int(clustersCount/2)
                if not killSmall or submethod != 'kmeans':
                    extraClusters = 0
                    # Small cluster killing is turned off when not using kmeans
                    killSmall = False
                # k-means # Find clusters of spectra
                if clustersCount == 1 or submethod == 'kmeans':
                    clusters = KMeans(n_clusters=clustersCount+extraClusters,
                                      n_jobs=-1, n_init=4
                                      ).fit(specFeat[d, :, :])
                else:
                    if submethod == 'birch':
                        clusters = Birch(n_clusters=clustersCount, threshold=0.01).fit(specFeat[d, :, :])
                    elif submethod == 'spectral':
                        clusters = SpectralClustering(n_clusters=clustersCount,
                                                      eigen_solver='arpack',
                                                      affinity='nearest_neighbors'
                                                      ).fit(specFeat[d, :, :])
                    elif submethod == 'ward':
                        clusters = AgglomerativeClustering(n_clusters=clustersCount+extraClusters, linkage='ward').fit(specFeat[d, :, :])
                    elif submethod == 'complete':
                        clusters = AgglomerativeClustering(n_clusters=clustersCount+extraClusters, linkage='complete').fit(specFeat[d, :, :])
                    elif submethod == 'average':
                        clusters = AgglomerativeClustering(n_clusters=clustersCount+extraClusters, linkage='average').fit(specFeat[d, :, :])
                    elif submethod == 'mini':
                        clusters = MiniBatchKMeans(n_clusters=clustersCount+extraClusters, batch_size=1000).fit(specFeat[d, :, :])
                # List of labels for each cluster, and corresponding cluster size (number of input spectra)
                labels, spectraPerCluster = np.unique(clusters.labels_, return_counts=True)

                if killSmall:
                    if dissolve:  # Dissolve small clusters and then merge each spectra with the closest cluster
                                  # Alternativelly merge clusters as a whole
                                  # Testing with synthetic spectra suggests dissolve works slightly better
                        for i in range(1, extraClusters+1):
                            # Find smaller cluster
                            smallerLabel = labels[np.argmin(spectraPerCluster)]
                            # Delete smaller cluster
                            clusters.cluster_centers_ = clusters.cluster_centers_[labels != smallerLabel]
                            spectraPerCluster = spectraPerCluster[labels != smallerLabel]
                            labels = labels[labels != smallerLabel]
                        # Reallocate spectra from dissolved clusters and get new cluster Labels for each spectrum
                        clusters.labels_ = clusters.predict(specFeat[d, :, :])
                        labels, spectraPerCluster = np.unique(clusters.labels_, return_counts=True)
                        # Update cluster centers
                        for label in labels:
                            clusters.cluster_centers_[label] = specFeat[d, clusters.labels_ == label, :].mean(axis=0)
                    else:
                        for i in range(1, extraClusters+1):
                            # Find smaller cluster
                            smallerIndex = np.argmin(spectraPerCluster)
                            smallerLabel = labels[smallerIndex]
                            smallerCount = spectraPerCluster[smallerIndex]
                            smallerCluster = clusters.cluster_centers_[smallerIndex]
                            # Merge smaller cluster with closest cluster (partner)
                            clusters.cluster_centers_ = clusters.cluster_centers_[labels != smallerLabel]
                            spectraPerCluster = spectraPerCluster[labels != smallerLabel]
                            labels = labels[labels != smallerLabel]
                            distances = (clusters.cluster_centers_ - smallerCluster)**2
                            distances = distances.sum(axis=1)
                            partner = np.argmin(distances)
                            clusters.labels_[clusters.labels_ == smallerLabel] = labels[partner]
                            # Merge clusters into new average
                            clusters.cluster_centers_[partner] = (spectraPerCluster[partner] * clusters.cluster_centers_[partner] + smallerCount * smallerCluster) / (spectraPerCluster[partner] + smallerCount)
                            spectraPerCluster[partner] += smallerCount

                inertia = np.zeros(fullSpectra.shape[-1])
                for binIndex in range(0, clustersCount):
                    espectros = fullSpectra[d, clusters.labels_ == binIndex, :] # To calculate the average spectra representative of each cluster, dont use the normalized specFeat spectra
                    center = espectros.mean(axis=0)  # Find average of spectra in cluster
                    inertia += ((espectros - center)**2).sum(axis=0)  # This is used to have diagnostics of the RMS error in the spectra classification
                    spectra[d, specIndex+binIndex, :] = center
                #print('spectra[0, :, :6].mean(axis=0)', spectra[0, :, :6].mean(axis=0))
                inertia = integrate.trapz(inertia, x=wavel)/(wavel[-1] - wavel[0])
                RMS[d, clustersCount-1] = np.sqrt(inertia / fullSpectra.shape[1])
                timePerCluster[d, specIndex:(specIndex+clustersCount)] = spectraPerCluster / speCount
                print('Number of clusters:', clustersCount, '   ', end="\r")

        np.savez_compressed(fname+'.timePerCluster', timePerCluster)
        fname += '.clusters'

    else:  # If not using machine learning clusters,
        # use Garcia's binning method based on the EPR650 criteria,
        # DOI: 10.1002/pip.2943
        for d in glo_dir:  # d = 1 for direct spectra
            # Sort spectra and integrated powers by EPR
            fullSpectra[d, :, :] = fullSpectra[d, np.argsort(EPR650[d, :]), :]
            P[d, :] = P[d, np.argsort(EPR650[d, :])]
            totalPower[d] = P[d, :].sum()  # Sum power
            binlimits.append([])
            binIndex = np.zeros(binMax, dtype=np.int)
            # Prepare data structure to store bin limits and averaged spectra.
            # numBins is total number of bins
            for numBins in bins:
                binlimits[d].append(np.zeros(numBins+1, dtype=np.int))
            accubin = 0
            # Calculate bin limits with equal power in each bin
            for specIndex in range(0, speCount):
                # Accumulate power and check if its a fraction of total power
                accubin += P[d, specIndex]
                for numBins in bins:
                    # check if power accumulated is a fraction of total power
                    if accubin >= (binIndex[numBins] + 1) * totalPower[d] / numBins - 0.0000001:  # This is needed to avoid rounding error
                        binIndex[numBins] += 1  # Create new bin if it is
                        binlimits[d][numBins - 1][binIndex[numBins]] = specIndex + 1  # Store bin limit
                        timePerCluster[d, getSpectrumIndex(numBins, binIndex[numBins] - 1)] = (specIndex + 1 - binlimits[d][numBins - 1][binIndex[numBins] - 1]) / speCount
            # Average spectra using the previously calculated bin limits

            for numBins in bins:  # iterate over every bin set
                specIndex = getSpectrumIndex(clustersCount, 0)

                inertia = np.zeros(fullSpectra.shape[-1])
                binlimits[d][numBins-1][-1] = speCount  # set the last bin limit to the total number of spectra
                for binIndex in range(0, numBins):
                    espectros = fullSpectra[d, binlimits[d][numBins-1][binIndex]:binlimits[d][numBins-1][binIndex+1]]
                    center = espectros.mean(axis=0)  # mean each bin
                    spectra[d, specIndex, :] = center
                    inertia += ((espectros - center)**2).sum(axis=0)
                    specIndex += 1
                inertia = integrate.trapz(inertia, x=wavel)/(wavel[-1] - wavel[0])
                RMS[d, numBins-1] = np.sqrt(inertia / fullSpectra.shape[1])

        np.savez_compressed(fname+'.timePerBin', timePerCluster)
        fname += '.bins'

    # speCount/attempts is the fraction of daytime hours in a year. It is needed to calculate the yearly averaged power yield including night time hours.
    spectra[0, 0, -1] = speCount/attempts
    print('Yearly daytime fraction is', speCount/attempts)
    np.savez_compressed(fname, spectra)

    if False:  # Set to True to plot RMS/inertia of each bin/cluster
        plt.figure()
        plt.xlabel('Number of spectra')
        plt.ylabel('RMS (W m$^{-2}$ nm$^{-1}$)')
        plt.xlim(0.1, binMax-0.3)
        plt.ylim(0.020, 0.105)
        ejes = plt.gca()
        ejes.xaxis.set_major_locator(plt.MultipleLocator(2))
        plt.tick_params(axis='y', right='on')
        if method == 'clusters':
            plt.scatter(bins, RMS[1], c='none', edgecolor='#0077BB', label='Clustering')  # Plot RMS for each cluster / bin for direct normal + circumsolar spectra
        else:
            plt.scatter(bins, RMS[1], c='#FF00FF', edgecolor='none', label='Binning')  # Plot RMS for each cluster / bin for direct normal + circumsolar spectra
        plt.legend(frameon=False)
        plt.savefig('RMS' + str(int(10*time.time())), dpi=300, bbox_inches="tight")
        plt.savefig('RMS' + str(int(10*time.time())) + '.svg', bbox_inches="tight")


def docs():  # Shows HELP file
    with open('./HELP', 'r') as fin:
        print(fin.read())
