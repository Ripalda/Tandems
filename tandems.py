# Random sampling of multijunction photovoltaic efficiencies. Jose M. Ripalda
# This script requires doing "pip install json_tricks" before running
# Tested with Python 2.7 and 3.6
# SMARTS 2.9.5 is required only to generate a new set of random spectra. 
# File "scs2.npy" can be used instead of SMARTS to load a set of binned spectra.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy import integrate
from scipy.interpolate import UnivariateSpline
import time
from matplotlib.colors import LinearSegmentedColormap
import scipy.constants as con
import pdb
import json
import json_tricks
import copy
import os.path
import subprocess as sub
from datetime import datetime
hc=con.h*con.c
q=con.e

np.set_printoptions(precision=3) # Print 4 decimal places only

colors = [(1, 0, 1), (0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]  # B -> G -> R
LGBT = LinearSegmentedColormap.from_list('LGBT', colors, 500)

def show_assumptions():
    EQE=np.exp(-((Energies-1.7)/2.2)**6)
    plt.figure()
    plt.xlim(0.4,4)
    plt.ylim(0,1)
    plt.title('Optimistic EQE model used in calculations \n Higher at all energies than any reported multijunction')
    plt.ylabel('External Quantum Efficiency')
    plt.xlabel('Photon energy (eV)')
    plt.plot(Energies,EQE)
    plt.show()
    
    #AOD random distribution used here is fitted to histograms by Jaus and Gueymard based on Aeronet data
    lrnd=np.random.lognormal(-2.141,1,size=1000000)
    lrnd[lrnd>2]=2 # For plotting purposes, AOD values are capped here at AOD=2, this is not done during calculations
    plt.figure()
    titel='AOD random distribution used here is fitted to \n histograms by Jaus and Gueymard based on Aeronet data, \n'
    titel+='DOI: 10.1063/1.4753849 \n In Jaus and Gueymard, percentiles at 25% and 75% are 0.06,0.23. \n'
    titel+='Here: '+str(np.percentile(lrnd, 25))+' '+str(np.percentile(lrnd, 75))
    plt.title(titel)
    plt.xlim(0,0.8)
    plt.xlabel('Aerosol Optical Depth at 500 nm')
    sali=plt.hist(lrnd,bins=200)
    plt.show()
    
    #PW random distribution used here is fitted to histograms by Jaus and Gueymard based on Aeronet data
    lrnd=0.39*np.random.chisquare(4.36,size=1000000)
    lrnd[lrnd>7]=7 # For plotting purposes, PW values are capped here at AOD=2, this is not done during calculations
    plt.figure()
    titel='PW random distribution used here is fitted to \n histograms by Jaus and Gueymard based on Aeronet data, \n'
    titel+='DOI: 10.1063/1.4753849 \n In Jaus and Gueymard, percentiles at 25% and 75% are 0.85,2.27. \n'
    titel+='Here: '+str(np.percentile(lrnd, 25))+' '+str(np.percentile(lrnd, 75))
    plt.title(titel)
    plt.xlim(0,6)
    plt.xlabel('Precipitable water (cm)')
    sali=plt.hist(lrnd,bins=40)
    plt.show()

# Load standard reference spectra ASTM G173
wavel,g1_5,d1_5=np.loadtxt("AM1_5 smarts295.ext.txt",delimiter=',',usecols=(0,1,2),unpack=True) 
Energies=1e9*hc/q/wavel # wavel is wavelenght in nm
a1123456789=[1,1,2,3,4,5,6,7,8,9]

# scs2.npy has been generated for Zenit<75.5 and -50<Lattitude<50
# scs.npy has been generated for Zenit<80 and -60<Lattitude<60
specs=np.load('scs2.npy') # Load binned spectra
specs[0,0,:]=g1_5
specs[1,0,:]=d1_5
Iscs=np.copy(specs)
P=np.zeros((2,46))

bindex=[]# arrays for spectral bin indexing
bindex.append([0])
k=1
for i in range(1,10): #bin set
    bindex.append([])
    for j in range(0,i): #bin counter
        bindex[i].append(k)        
        k+=1
# These arrays are used to speed up calculations by limiting the energy gap search space.
# Initial guess at where the eff maxima are. 
# The search space is expanded as needed if high eff are found at the edges of the search space. 
Emax=[] # Max energy for each gap
Emin=[] # Min energy for each gap
Emax.append([1.6]) # 1 junction
Emin.append([0.9])
Emax.append([1.40,2.00]) # 2  junctions
Emin.append([0.85,1.40])
Emax.append([1.20,1.67,2.00]) # 3  junctions
Emin.append([0.65,1.15,1.60])
Emax.append([1.10,1.51,1.90,2.10]) # 4  junctions
Emin.append([0.50,0.94,1.25,1.80])
Emax.append([0.95,1.15,1.55,2.00,2.25]) # 5  junctions
Emin.append([0.50,0.83,1.15,1.50,1.90])
Emax.append([0.95,1.15,1.50,1.90,2.10,2.30]) # 6  junctions
Emin.append([0.50,0.78,1.05,1.40,1.70,2.05])

# These arrays are used to discard band gap combinations that are too unevenly spaced in energy
maxDif=[]
minDif=[] 
maxDif.append([1.00]) # 2 junctions, max E difference between junctions
minDif.append([0.70]) # 2 junctions, min E difference between junctions
maxDif.append([0.65,0.55])
minDif.append([0.50,0.50]) # 3 junctions
maxDif.append([0.60,0.55,0.65])
minDif.append([0.45,0.45,0.55]) # 4 junctions
maxDif.append([0.50,0.60,0.45,0.55])
minDif.append([0.30,0.40,0.40,0.45]) # 5 junctions
maxDif.append([0.50,0.50,0.40,0.40,0.50])
minDif.append([0.28,0.28,0.32,0.32,0.42]) # 6 junctions

#Varshni, Y. P. Temperature dependence of the energy gap in semiconductors. Physica 34, 149 (1967)
def Varshni(T): #Gives gap correction in eV relative to 300K using GaAs parameters. T in K
    return (T**2)/(T+572)*-8.871e-4+0.091558486 # GaAs overestimates effect for most other semiconductors. Effect is small anyway.

class effis(object): 
    """ Class to hold results sets of yearly average photovoltaic efficiency """  
    # s = self = current object instance
    ERE=0.01 #external radiative efficiency without mirror. With mirror ERE increases by a factor (1 + beta)
    beta=11 #n^2 squared refractive index = radiative coupling parameter = substrate loss.
    rgaps=0 # Results Array with high efficiency Gap combinations found by trial and error
    Is=0 # Results Array with Currents as a function of the number of spectral bins, 0 is standard spectrum 
    effs=0 # Results Array with Efficiencies as a function of the number of spectral bins, 0 is standard spectrum
    gaps=[0,0,0,0,0,0] # If a gap is 0, it is randomly chosen by tandems.sample(), otherwise it is kept fixed at value given here.
    auxEffs=0 # Aux array for efficiencies. Has the same shape as rgaps for plotting and array masking. 
    auxIs=0 # Aux array for plotting. sum of short circuit currents from all terminals.
    bins=6 # bins is number of spectra used to evaluate eff, an array can be used to test the effect of the number of spectral bins
    # See convergence=True. Use [4] or more bins if not testing for convergence as a function of the number of spectral bins 
    convergence=False # Set to True to test the effect of changing the number of spectral bins  used to calculate the yearly average efficiency
    Irc=0 # Radiative coupling current
    Itotal=0 # Isc
    Pout=0 # Power out
    concentration=1000
    transmission=0.02 # Subcell thickness cannot be infinite, 3 micron GaAs has transmission in the 2 to 3 % range (depending on integration range)
    thinning=False # Automatic subcell thinning for current matching
    thinSpec=1 # Spectrum used to calculate subcell thinning for current matching. Integer index in specs array. 
    thinTrans=1 # Array with transmission of each subcell
    effMin=0.02 # Lowest sampled efficiency value relative to maximum efficiency. Gaps with lower efficiency are discarded.
    d=0 # 0 for global spectra, 1 for direct spectra
    Tmin=15+273.15 # Minimum ambient temperature at night in K
    deltaT=np.array([30,55]) # Device T increase over Tmin caused by high irradiance (1000 W/m2), first value is for flat plate cell, second for high concentration cell
    # T=70 for a 1mm2 cell at 1000 suns bonded to copper substrate. Cite I. Garcia, in CPV Handbook, ed. by: I. Rey-Stolle, C. Algora
    junctions=6
    topJunctions=6 # Number of series conected juctions in top stack (topJunctions=junctions in 2 terminal devices)
    name='Test' # use for file saving
    cells=1000 # Desired number of calculated tandem cells
    # Total series resistance of each series connected stack in Ohm*m2
    R=5e-7 # Default is optimistic value for high concentration devices
    # R=4e-5 is suggested for one sun flat plate devices
    EQE=0 # This is changed in __init__, type show_assumptions() to see actual EQE
    Ijx=0 # Array with the external photocurrents integrated from spectrum. Is set by getIjx()
    T=0 # set from irradiance at run time
    def __init__(s, **kwargs):
        s.EQE=np.exp(-((Energies-1.7)/2.2)**6) # Optimistic default EQE model, 74% at 3.5 eV, >99% from 2.5 to 0.9 eV, 96% at 0.4 eV
        # can do effis( EQE = 1 ) to override this default. Any array with the same length as the spectra will do too
        for k, v in kwargs.items():
            setattr(s, k, v)
        if type(s.bins)==int:
            s.bins=[s.bins]
        def integra(d,spec): # Function to Integrate spectra from UV to given wavelength 
            global Iscs, P
            P[d,spec]=integrate.trapz(specs[d,spec,:], x=wavel) # Power per unit area ( W / m2 )
            Iscs[d,spec,:]=(q/hc)*np.insert(1e-9*integrate.cumtrapz(s.EQE*specs[d,spec,:]*wavel,x=wavel),0,0) # Current per unit area ( A / m2 ), wavelength in nm
        for d in [0,1]: # Integrate spectra from UV to given wavelength 
            for i in range(0,46):
                integra(d,i)
        if s.topJunctions==0:
            s.topJunctions=s.junctions
        if s.convergence:
            s.bins=[1,2,3,4,5,6,7,8,9]
        if s.concentration>1:
            s.d=1 # use direct spectra
            
    def intSpec(s,energy,spec): # Returns integrated photocurrent from given photon energy to UV
        return np.interp(1e9*hc/energy/q,wavel,Iscs[s.d,spec,:]) # interpolate integrated spectra. Wavelength in nm
    def getIjx(s,spec): # Get current absorbed in each junction, external photocurrent only
        if spec!=None: # if None keep using previous results for Ijx
            s.Ijx=np.zeros(s.junctions)
            IfromTop=0
            upperIUVE=0
            for i in range(s.junctions-1,-1,-1): # From top to bottom: get external photocurrent in each junction
                IUVE=s.intSpec(s.gaps[i]+Varshni(s.T),spec) # IUVE = Integrated current from UV to given Energy gap
                Ijx0=s.concentration*(IUVE-upperIUVE) # Get external I per junction 
                #print (i,s.gaps[i],IUVE,IUVE-previousIUVE)
                upperIUVE=IUVE
                if i!=0:
                    s.Ijx[i]=(1-s.transmission)*Ijx0+IfromTop # Subcell thickness cannot be infinite, 3 micron GaAs has transmission in the 2 to 3 % range (depending on integration range)
                    IfromTop=s.transmission*Ijx0
                else:
                    s.Ijx[i]=Ijx0+IfromTop # bottom junction does not transmit (back mirror)
    def thin(s,topJ,bottomJ): # Calculate transmission factors for each subcell in series to maximize current under spectrum given in thinSpec      
        # Top cell thinning: 
        # - From top to bottom junction
        #       - If next current is lower :
        #             - Find average with next junction
        #             - If average I is larger than next junction I, extend average and repeat this step
                
        s.getIjx(s.thinSpec) # get external photocurrent  
        initialIjx = np.copy(s.Ijx)
        
        Ijxmin=0
        while int(Ijxmin*100) != int(s.Ijx.min()*100): # While min I keeps going up
            Ijxmin = s.Ijx.min()
            i = topJ
            while i > bottomJ-1 : # Spread I from top to bottom
                if s.Ijx[i] > s.Ijx[i-1]: # If next current is lower 
                    imean=i
                    mean=s.Ijx[i]
                    previousMean=0
                    while mean > previousMean :
                        imean-=1
                        previousMean=mean
                        if imean > -1 :
                            mean=np.mean(s.Ijx[imean:i+1])
                    s.Ijx[imean:i+1] = mean
                    i = imean
                i-=1

        s.thinTrans=initialIjx/s.Ijx

        #print ('in',initialIjx,initialIjx.sum())
        #print ('out',s.Ijx,s.Ijx.sum())
        #print (s.thinTrans)
        
    def serie(s,topJ,bottomJ): # Get power from series connected subcells with indexes topJ to bottomJ. topJ=bottomJ is single junction. 
        # 1 - Get external photocurrent in each junction
        # 2 - Get current at the maximum power point from min external photocurrent
        # 3 - Add radiative coupling, recalculate maximum power point, this changes radiative coupling, repeat until self consistency
        # 4 - Calculate power out
        
        # Do "tandems.show_assumptions()" to see EQE model used and some characteristics of the spectral set used.

        if (topJ<0):
            return
        kT=con.k*s.T
        Irc0=s.Irc #radiative coupling from upper stack 


        Ijx=s.Ijx*s.thinTrans # thinTrans is set by thin()
        Imax=Ijx[bottomJ:topJ+1].min() # Initial guess for current at the maximum power point 
        Imaxs=[0,Imax]
        while (((Imax-Imaxs[-2])/Imax)**2)>1e-7: # Loop to s consistently refine max power point
            V=0
            s.Irc=Irc0 # Radiative coupling from upper stack 
            Ij=np.copy(Ijx) # Current absorbed in each junction
            for i in range(topJ,bottomJ-1,-1): # From top to bottom: get photocurrent in each junction including radiative coupling
                Ij[i]+=s.Irc # Include emitted light from upper junction
                if (Ij[i]>Imax): # If there is excess current in this junction, radiative coupling
                    s.Irc=s.beta*s.ERE*(Ij[i]-Imax) #radiative coupling 
                else:
                    s.Irc=0
            Ijmin=Ij[bottomJ:topJ+1].min() # Min current in series connected stack
            I=Ijmin*np.arange(0.8,1,0.0001) # IV curve sampling   
            for i in range(topJ,bottomJ-1,-1): # From top to bottom: Sample IV curve, get I0
                if (i==bottomJ): # The bottom junction of each series connected stack has some additional photon recycling due to partial back reflection of luminescence
                    backLoss=1 # This is the loss due to an air gap between mechanically stacked cells. Same loss assumed for back contact metal mirrors.
                else:
                    backLoss=s.beta
                I0=(1+backLoss) * 2*np.pi*q * np.exp(-1*s.gaps[i]*q/kT) * kT**3*((s.gaps[i]*q/kT+1)**2+1) / (con.h*hc**2) / s.ERE # Dark current at V=0 in A / m2 s
                V+=(kT/q)*np.log((Ij[i]-I)/I0+1) # add voltage of series connected cells
            V-=s.R*I # V drop due to series resistance
            Imax=I[np.argmax(I*V)] # I at max power point
            Imaxs.append(Imax)
        if len(Imaxs)>10:
            print ('s consistency is slowing convergence while finding the maximum power point.')
            print ('ERE or beta might be too high.')
            print ('Current at the maximum power point is converging as:', Imaxs)
            pdb.set_trace()
        s.Itotal+=Ijmin
        s.Pout+=(I*V).max()
        return
        
    def stack(s,spec): # Use a single spectrum to get power from 4 terminal tandem, if topJunctions=junctions the result is for 2 terminal tandem.
        s.Irc=0 # For top cell there is no radiative coupling from upper cell
        s.T=s.Tmin+s.deltaT[s.d]*P[s.d,spec]/1000 # To a first approximation, cell T is a linear function of irradiance.
        # T=70 for a 1mm2 cell at 1000 suns bonded to copper substrate. Cite I. Garcia, in CPV Handbook, ed. by: I. Rey-Stolle, C. Algora
        s.getIjx(spec) # Get external photocurrents
        s.serie(s.junctions-1,s.junctions-s.topJunctions) # Add efficiency from top stack, topJunctions is number of juntions in top stack
        s.serie(s.junctions-s.topJunctions-1,0) # Add efficiency from bottom stack
        return
    def sample(s): # Sample efficiencies for random band gaps. The multijunction type is defined with junctions,topJunctions
        startTime=time.time()
        ncells=0 # Number of calculated gap combinations
        nres=0
        effmax=0
        Emin_=np.array(Emin[s.junctions-1]) # Initial guess at where the eff maxima are. 
        Emax_=np.array(Emax[s.junctions-1]) # The search space is expanded as needed if high eff are found at the edges of the search space.        
        minDif_=np.array(minDif[s.junctions-2])
        maxDif_=np.array(maxDif[s.junctions-2])
        # REMOVE SEED unless you want to reuse the same sequence of random numbers (e.g.: compare results after changing one parameter)
        #np.random.seed(07022015)
        s.rgaps=np.zeros((s.cells+1000,s.junctions)) # Gaps
        s.auxIs=np.zeros((s.cells+1000,s.junctions)) # Aux array for plotting only
        s.auxEffs=np.zeros((s.cells+1000,s.junctions)) # Aux array for plotting only  
        s.Is=np.zeros((s.cells+1000,10)) # Currents as a function of the number of spectral bins, 0 is standard spectrum 
        s.effs=np.zeros((s.cells+1000,10)) # Efficiencies as a function of the number of spectral bins, 0 is standard spectrum 
        fixedGaps=np.copy(s.gaps) # Copy input gaps to remember which ones are fixed, if gap==0 make it random
        while (nres<s.cells+1000): # Loop to randomly sample a large number of gap combinations
            s.gaps=np.zeros(s.junctions)  
            lastgap=0
            i=0
            while i<s.junctions:     # From bottom to top: define random gaps
                if i>0:
                    Emini = max(Emin_[i] , lastgap + minDif_[i-1]) # Avoid gap combinations that are too unevenly spaced
                    Emaxi = min(Emax_[i] , lastgap + maxDif_[i-1])
                else:
                    Emini = Emin_[i]
                    Emaxi = Emax_[i]
                Erange = Emaxi - Emini
                if fixedGaps[i]==0:
                    s.gaps[i] = Emini + Erange * np.random.rand() #define random gaps
                else:
                    s.gaps[i]=fixedGaps[i] # Any gaps with a value other than 0 are kept fixed, example: tandems.effis(gaps=[0.7,0,0,1.424,0,2.1]) .
                if i>0 and fixedGaps.sum(): # If there are fixed gaps check gaps are not too unevenly spaced
                    if not ( minDif_[i-1] < s.gaps[i]-lastgap < maxDif_[i-1] ): # gap difference is not in range
                        lastgap=0 # Discard gaps and restart if gaps are too unevenly spaced
                        i=0
                    else:
                        lastgap=s.gaps[i]
                        i+=1
                else:
                    lastgap=s.gaps[i]
                    i+=1
                    
            if s.thinning:
                
                bottomJts=s.junctions-s.topJunctions # index number for the bottom junction of the top stack
                s.thin(s.junctions-1,bottomJts) # Find optimal subcell thinning for top stack for a certain spectrum
                
                if bottomJts>0: # if there is a lower stack
                    exIb=s.Ijx[bottomJts]-s.Ijx[bottomJts:].min() # The excess current from the bottom junction of the top stack, goes to lower stack
                    s.Ijx[bottomJts]-=exIb
                    s.Ijx[bottomJts-1]+=exIb
                    s.thin(bottomJts-1,0,None) # Find optimal subcell thinning for bottom stack. if None: use the previously specified spectrum 
                
            s.Itotal=0
            s.Pout=0
            s.stack(1) # calculate power out with average spectrum (1 bin).
            eff=s.Pout/P[s.d,1]/s.concentration # get efficiency
            if (eff>effmax-s.effMin-0.01): # If gap combination is good, do more work on it
                if int(100000*time.time() % 1000)==1: # Show calculation progress from time to time
                    print ('Tried',ncells,', got ',nres,' candidate gap combinations.')
                for gi,gap in enumerate(s.gaps): # Expand edges of search space if any of the found gaps are near an edge
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
                    effmax=eff
                s.rgaps[nres,:]=s.gaps
                s.effs[nres,1]=eff 
                s.Is[nres,1]=s.Itotal 
                for i in s.bins: # loop for number of bins
                    s.Itotal=0
                    Pin=0
                    s.Pout=0                  
                    for j in range(0,a1123456789[i]): # loop for bin index.    a1123456789=[1,1,2,3,4,5,6,7,8,9]
                        s.stack(bindex[i][j]) # calculate power out for each spectra and add to Pout to calculate average efficiency
                        Pin+=P[s.d,bindex[i][j]]
                    s.effs[nres,i]=s.Pout/Pin/s.concentration
                    s.auxEffs[nres,:]=np.zeros(s.junctions)+s.effs[nres,i]
                    s.auxIs[nres,:]=np.zeros(s.junctions)+s.Itotal/a1123456789[i] # a1123456789=[1,1,2,3,4,5,6,7,8,9]
                    s.Is[nres,i]=s.Itotal/a1123456789[i]
                nres+=1
                                
            ncells+=1
            
        print ('Emin',Emin_)
        print ('Emax',Emax_)
        print ('maxDif_',maxDif_)
        print ('minDif_',minDif_)
            
        mask = s.auxEffs > s.auxEffs.max()-s.effMin
        s.rgaps = s.rgaps[mask] # Discard results below the efficiency threshold set by effMin
        s.auxIs = s.auxIs[mask] # As a side effect of this cut off, arrays are flattened
        s.auxEffs = s.auxEffs[mask] # [:,0]
        threshold = s.effs[:,s.bins[-1]].max()-s.effMin
        mask = s.effs[:,s.bins[-1]] > threshold
        s.effs = s.effs[mask]
        s.Is = s.Is[mask]
        tiempo=int(time.time()-startTime)
        res=np.size(s.rgaps)/s.junctions
        print ('Calculated ',ncells, ' and saved ', res, ' gap combinations in ',tiempo,' s :',res/(tiempo+1),' results/s')
    def plot(s):
        startTime=time.time()
        print ('I min, I max : ',s.auxIs.min(),s.auxIs.max())
        print ('eff min, eff max : ',s.auxEffs.min(),s.auxEffs.max())
        res=np.size(s.rgaps)/s.junctions
        
        #rgaps=s.rgaps.flatten() 
        #auxEffs=s.auxEffs.flatten()
        #rIs_=(s.auxIs[:,0]-s.auxIs[:,0].min())/(s.auxIs[:,0].max()-s.auxIs[:,0].min())
        Is_=np.copy(s.auxIs)
        Is_=(Is_-Is_.min())/(Is_.max()-Is_.min())
        srIs_=(s.Is[:,s.bins[-1]]-s.Is[:,s.bins[-1]].min())/(s.Is[:,s.bins[-1]].max()-s.Is[:,s.bins[-1]].min())
        if s.convergence:
            plt.figure()
            plt.xlabel('Number of spectral bins')
            plt.ylabel('Absolute efficiency change \n by increasing number of bins (%)')
            for i in range(0,int(res)):
                diffs=[]
                for j in s.bins[:-1]:
                    diffs.append(100*(s.effs[i,j+1]-s.effs[i,j]))
                plt.plot(s.bins[:-1],diffs,color=LGBT(srIs_[i]),linewidth=0.1) #  color=LGBT(auxIs[i*s.junctions])
            plt.savefig('lat50/Convergence '+s.name+' '+str(int(s.junctions))+' '+str(int(s.topJunctions))+' '+str(int(s.concentration))+' '+str(int(startTime)),dpi=600)
            plt.show()
        plt.figure()
        plt.ylim(100*(s.effs[:,s.bins[-1]].max()-s.effMin),100*s.effs[:,s.bins[-1]].max()+0.1)
        plt.xlim(0.5,2.5)
        #plt.grid(True)
        plt.minorticks_on()
        plt.tick_params(direction='out',which='minor')
        plt.tick_params(direction='inout',pad=6)
        plt.ylabel('Yearly averaged efficiency (%)')
        plt.xlabel('Energy gaps (eV)')
        plt.plot([0.50,0.50],[0,100],c='grey',linewidth=0.5)
        plt.text(0.5,100*s.auxEffs.max()+.2,'A')
        plt.plot([0.69,0.69],[0,100],c='grey',linewidth=0.5)
        plt.text(0.69,100*s.auxEffs.max()+.2,'B')
        plt.plot([0.92,0.92],[0,100],c='grey',linewidth=0.5)
        plt.text(0.92,100*s.auxEffs.max()+.2,'C')
        plt.plot([1.1,1.1],[0,100],c='grey',linewidth=0.5)
        plt.text(1.1,100*s.auxEffs.max()+.2,'D')
        plt.plot([1.33,1.33],[0,100],c='grey',linewidth=0.5)
        plt.text(1.33,100*s.auxEffs.max()+.2,'E1')
        plt.plot([1.63,1.63],[0,100],c='grey',linewidth=0.5)
        plt.text(1.63,100*s.auxEffs.max()+.2,'E2')
        plt.scatter(s.rgaps,100*s.auxEffs,c=Is_,s=70000/len(s.effs[:,s.bins[-1]]), edgecolor='none', cmap=LGBT)

        #if not s.convergence:
        plt.savefig('lat50/'+s.name+' '+str(int(s.junctions))+' '+str(int(s.topJunctions))+' '+str(int(s.concentration))+' '+str(int(startTime)),dpi=600)
        plt.show()        
        plt.xlim(100*(s.auxEffs.max()-s.effMin),100*s.auxEffs.max())
        plt.xlabel('Yearly averaged efficiency (%)')
        plt.ylabel('Count')
        plt.hist(100*s.effs[:,s.bins[-1]], bins=30)
        #if not s.convergence:
        plt.savefig('lat50/Hist Eff '+s.name+' '+str(int(s.junctions))+' '+str(int(s.topJunctions))+' '+str(int(s.concentration))+' '+str(int(startTime)),dpi=600)      
        plt.show()
    def save(s):
        with open(s.name+' '+str(int(s.junctions))+' '+str(int(s.topJunctions))+' '+str(int(s.concentration))+' '+str(int(time.time())), "w") as f:
            f.write(json_tricks.dumps(s))
    def load(s,fname):
        with open(fname,"r") as f:
            t0=f.readlines()
            t0=''.join(t0)
            return json_tricks.loads(t0)
            
def generate_spectral_bins():
    # NECESARY CHANGES IN SMARTS 2.9.5 SOURCE CODE
    # Line 189
    #       batch=.TRUE.
    # Line 1514
    #      IF(Zenit.LE.75.5224)GOTO 13
    #      WRITE(16,103,iostat=Ierr24)Zenit
    # 103  FORMAT(//,'Zenit = ',F6.2,' is > 75.5224 deg. (90 in original code) This is equivalent to AM < 4'
    # This change is needed because trackers are shadowed by neighboring trackers when the sun is near the horizon. 
    # Zenit 80 is already too close to the horizon to use in most cases due to shadowing issues.

    numres=20000 # number of generated spectra

    directos=np.zeros((numres,2002))
    globales=np.zeros((numres,2002))
    EPR650d=np.zeros(numres) # Ivan Garcia's criteria for binning, store EPR value for each spectrum
    EPR650g=np.zeros(numres) # Ivan Garcia's criteria for binning
    Pd=np.zeros(numres) # Power for each spectrum
    Pg=np.zeros(numres)

    #os.chdir('/home/jose/SMARTS')
    with open ('smarts295.in_.txt', "r") as fil: # Load template for SMARTS input file
        t0=fil.readlines()
    t0=''.join(t0)
    sc=0
    for i in range(0,numres):
        tama=0
        while tama==0: # Generate random time, location, AOD, PW
            t1=t0.replace('AOD#',str(min(5.4,np.random.lognormal(-2.141,1)))) # Substitute AOD and PW in template file
            t1=t1.replace('PW#',str(min(12,0.39*np.random.chisquare(4.36)))) # For more info do "tandems.show_assumptions()"
            t2=datetime.fromtimestamp(3503699364*np.random.rand()+1503699364).strftime("%Y %m %d %H")+' '
            t2+='%.2f' % (100*np.arccos(2*np.random.rand()-1)/np.pi-50)+' '  # 50 > Latitude > -50
            t2+='%.2f' % (360*np.random.rand()-180)+' 0\t\t\t!Card 17a Y M D H Lat Lon Zone\n\n'
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
            sc+=1
            ou=sub.check_call('./smartsAM4', shell=True) # execute SMARTS
            tama=os.stat('smarts295.ext.txt').st_size # check output file
            if tama>0:
                try: # load output file if it exists
                    wavel,g1,d1=np.loadtxt("smarts295.ext.txt",delimiter=' ',usecols=(0,1,2),unpack=True,skiprows=1)#Global Tilted = 1,Direct Normal= 2, Diffuse Horizontal=3
                except:
                    tama=0
        globales[i,:]=g1
        directos[i,:]=d1
        Pg[i]=integrate.trapz(g1, x=wavel)
        Pd[i]=integrate.trapz(d1, x=wavel)
        EPR650g[i]=integrate.trapz(g1[:490], x=wavel[:490])
        EPR650g[i]=(Pg[i]-EPR650g[i])/EPR650g[i] # Calculate EPR, Ivan Garcia's criteria for binning
        EPR650d[i]=integrate.trapz(d1[:490], x=wavel[:490])
        EPR650d[i]=(Pd[i]-EPR650d[i])/EPR650d[i]
    print ('Called smarts '+str(sc)+' times')

    glob=globales[np.argsort(EPR650g),:] # Sort spectra and integrated powers by EPR
    dire=directos[np.argsort(EPR650d),:]
    Pgs=Pg[np.argsort(EPR650g)]
    Pds=Pd[np.argsort(EPR650d)]
    # global spectra
    gbinsize=Pgs.sum() # Sum power
    gbincount=np.zeros(9)
    gbinlimits=[]
    gbinspecs=[]
    for i in range(0,9): # Prepare data structure to store bin limits and averaged spectra
        gbinlimits.append(np.zeros(i+2))
        gbinspecs.append(np.zeros((i+1,2002)))
    # direct spectra        
    dbinsize=Pds.sum() # Sum power
    dbincount=np.zeros(9)
    dbinlimits=[]
    dbinspecs=[]
    for i in range(0,9): # Prepare data structure to store bin limits and averaged spectra
        dbinlimits.append(np.zeros(i+2))
        dbinspecs.append(np.zeros((i+1,2002)))
    # global spectra        
    accubin=0 # Calculate bin limits with equal power in each bin
    for i in range(0,numres):
        accubin+=Pgs[i]
        for j in range(0,9):
            if accubin>=gbinsize/(j+1)*(gbincount[j]+1):
                gbincount[j]+=1
                gbinlimits[j][gbincount[j]]=i+1 # Calculate bin limits
    # direct spectra
    accubin=0 # Calculate bin limits with equal power in each bin
    for i in range(0,numres):
        accubin+=Pds[i]
        for j in np.arange(0,9,1):
            if accubin>=dbinsize/(j+1)*(dbincount[j]+1):
                dbincount[j]+=1
                dbinlimits[j][dbincount[j]]=i+1 # Calculate bin limits
                
    for i in range(0,9): #iterate over every bin set
        gbinlimits[i][-1]=numres # set the last bin limit to the total number of spectra
        dbinlimits[i][-1]=numres
        # Average spectra using the previously calculated bin limits
    for i in range(0,9): #bin set
        for j in np.arange(0,i+1,1): #bin counter
            gbinspecs[i][j]=np.sum(glob[gbinlimits[i][j]:gbinlimits[i][j+1]],axis=0)/(gbinlimits[i][j+1]-gbinlimits[i][j]) #/np.sum(Pgs[gbinlimits[i][j]:gbinlimits[i][j+1]])
    for i in range(0,9): #bin set
        for j in range(0,i+1): #bin counter
            dbinspecs[i][j]=np.sum(dire[dbinlimits[i][j]:dbinlimits[i][j+1]],axis=0)/(dbinlimits[i][j+1]-dbinlimits[i][j]) #/np.sum(Pds[dbinlimits[i][j]:dbinlimits[i][j+1]])

    Iscs=np.zeros((2,46,2002)) # Prepare data structure for saving to file

    k=1
    for i in range(0,9): #bin set
        for j in range(0,i+1): #bin counter
            Iscs[0,k,:]=gbinspecs[i][j]
            Iscs[1,k,:]=dbinspecs[i][j]
            k+=1
    np.save('scs2', Iscs)
