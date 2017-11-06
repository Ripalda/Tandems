# Random sampling of multijunction photovoltaic efficiencies. Jose M. Ripalda
# This script requires doing "pip install json_tricks" before running
# SMARTS 2.9.5 is required only to generate a new set of random spectra. 
# File "scs.npy" can be used instead of SMARTS to load a set of binned spectra.
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

colors = [(1, 0, 1), (0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]  # B -> G -> R
LGBT = LinearSegmentedColormap.from_list('LGBT', colors, 500)

# Load standard reference spectra ASTM G173
wavel,g1_5,d1_5=np.loadtxt("AM1_5 smarts295.ext.txt",delimiter=',',usecols=(0,1,2),unpack=True) 
Energies=1e9*hc/q/wavel # wavel is wavelenght in nm
a1123456789=[1,1,2,3,4,5,6,7,8,9]

def show_assumptions():
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

specs=np.load('scs.npy') # Load binned spectra
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
        
Ef=[] # These arrays are used to speed up calculations 
Es=[] # by limiting the energy gap search space.
# initial guess at where the eff maxima are. The search space is expanded as needed if high eff are found at the edges of the search space.  
Ef.append([])# 0 junctions, padding
Es.append([])
Es.append([.9]) # First set of values are base energy shifts between junctions
Es.append([.6,.7])
Es.append([.55,.50,.55])
Es.append([.55,.45,.45,.55])
Es.append([.55,.3,.4,.4,.5])
Es.append([.55,.3,.3,.35,.35,.5])
Ef.append([.8])# 1 junction. Second set of values are scatter factors for the energy shifts
Ef.append([.7,.4])# 2 junctions
Ef.append([.4,.25,.1])# 3 junctions
Ef.append([.4,.25,.1,.1])# 4 junctions
Ef.append([.4,.25,.25,.1,.1])# 5 junctions
Ef.append([.4,.25,.25,.1,.1,.1])# 6 junctions
#Varshni, Y. P. Temperature dependence of the energy gap in semiconductors. Physica 34, 149 (1967)
def Varshni(T): #Gives gap correction in eV relative to 300K using GaAs parameters. T in K
    return (T**2)/(T+572)*-8.871e-4+0.091558486 # GaAs overestimates effect for most other semiconductors. Effect is small anyway.

class effis(object): 
    """ Class to hold results sets of yearly average photovoltaic efficiency """  
    ERE=0.01 #external radiative efficiency without mirror. With mirror ERE increases by a factor (1 + beta)
    beta=11 #n^2 squared refractive index = radiative coupling parameter = substrate loss.
    rgaps=0 # Array with many Gap combinations
    gaps=0 # 
    auxIs=0 # Aux array for sum of short circuit currents from all terminals.
    auxeffs=0 # Aux array for efficiencies. Has the same shape as rgaps for plotting and array masking.
    Is=0 # Currents as a function of the number of spectral bins, 0 is standard spectrum 
    effs=0 # Efficiencies as a function of the number of spectral bins, 0 is standard spectrum 
    numbins=[4] # numbins is number of spectra used to evaluate eff, an array can be used to test the effect of the number of spectral bins
    # See convergence=True. Use [4] or more bins if not testing for convergence as a function of the number of spectral bins 
    convergence=False # Set to True to test the effect of changing the number of spectra used to calculate the yearly average efficiency
    Irc=0 # Radiative coupling current
    Itotal=0 # Isc
    Pout=0 # Power out
    concentration=1000
    thinning=False # Automatic top cell thinning for current matching
    effmin=0.02 # Lowest sampled efficiency value relative to maximum efficiency. Gaps with lower efficiency are discarded.
    d=0
    Tmin=15+273.15 # Minimum ambient temperature at night in K
    deltaT=np.array([30,55]) # Device T increase over Tmin caused by high irradiance (1000 W/m2), first value is for flat plate cell, second for high concentration cell
    # T=70 for a 1mm2 cell at 1000 suns bonded to copper substrate. Cite I. Garcia, in CPV Handbook, ed. by: I. Rey-Stolle, C. Algora
    junctions=3
    numTop=0 # Number of series conected juctions in top stack (numTop=junctions in 2 terminal devices)
    name='Test' # use for file saving
    cells=1000 # Desired number of calculated tandem cells
    # Total series resistance of each series connected stack in Ohm*m2
    R=5e-7 # Default is optimistic value for high concentration devices
    # R=4e-5 is suggested for one sun flat plate devices
    EQE=0 # This is changed in __init__, type show_assumptions() to see actual EQE
    def __init__(self, **kwargs):
        self.EQE=np.exp(-((Energies-1.7)/2.2)**6) # Optimistic default EQE model, 74% at 3.5 eV, >99% from 2.5 to 0.9 eV, 96% at 0.4 eV
        # can do effis( EQE = 1 ) to override this default. Any array with the same length as the spectra will do too
        for k, v in kwargs.items():
            setattr(self, k, v)
        def integra(d,spec): # Integrate spectra
            global Iscs, P
            P[d,spec]=integrate.trapz(specs[d,spec,:], x=wavel) # W / m2
            Iscs[d,spec,:]=(q/hc)*np.insert(1e-9*integrate.cumtrapz(self.EQE*specs[d,spec,:]*wavel,x=wavel),0,0) # Current in A / m2, wavelength in nm
        for d in [0,1]: # Integrate spectra
            for i in range(0,46):
                integra(d,i)
        if self.numTop==0:
            self.numTop=self.junctions
        if self.convergence:
            self.numbins=[1,2,3,4,5,6,7,8,9]
        if self.concentration>1:
            self.d=1 # use direct spectra
            
    def Isce(self,gap,spec): # Integrated Current as function of gap
        return np.interp(1e9*hc/gap/q,wavel,Iscs[self.d,spec,:]) # wavelength in nm
    
    def serie(self,topJ,bottomJ,spec): # Use a single spectrum to get power from series connected subcells with indexes topJ to bottomJ. topJ=bottomJ is single junction. 
        # PSEUDOCODE:
        # get external photocurrent in each junction
        # get current at the maximum power point from min external photocurrent
        # add radiative coupling, recalculate maximum power point, this changes radiative coupling, repeat until self consistency
        # calculate power out
        
        # Do "tandems.show_assumptions()" to see EQE model used and some characteristics of the spectral set used.

        if (topJ<0):
            return
        T=self.Tmin+self.deltaT[self.d]*P[self.d,spec]/1000 # To a first approximation, cell T is a linear function of irradiance.
        # T=70 for a 1mm2 cell at 1000 suns bonded to copper substrate. Cite I. Garcia, in CPV Handbook, ed. by: I. Rey-Stolle, C. Algora
        kT=con.k*T
        Irc0=self.Irc #radiative coupling from upper stack 
        Ijx=np.zeros(self.junctions) # Current absorbed in each junction, external photocurrent only
        for i in range(topJ,bottomJ-1,-1): # From top to bottom: get external photocurrent in each junction
            Ijx[i]=self.concentration*(self.Isce(self.gaps[i]+Varshni(T),spec)-self.Isce(self.gaps[i+1]+Varshni(T),spec)) # Get external I per junction 
        if self.thinning:
            # Top cell thinning: 
            # 1 - Find min I,
            # 2 - average 2,3,4,5 adjacent upper junctions, choose the higher I,
            # 3 - repeat until min I does not change
            imi=np.argmin(Ijx)
            if imi<topJ:
                Ijxmin=0
                while Ijx.min()<>Ijxmin:
                    Ijxmin=Ijx.min()
                    mav=[]
                    for i in range(imi+1,topJ+1):
                        mav.append(np.mean(Ijx[imi:i]))
                    mav=np.array(mav)
                    imavmax=np.argmax(mav)
                    for i in range(imi,imavmax+1):
                        Ijx[i]=mav.max()
            
        Imax=Ijx[bottomJ:topJ+1].min() # Initial guess for current at the maximum power point 
        Imaxs=[0,Imax]
        while (((Imax-Imaxs[-2])/Imax)**2)>1e-7: # Loop to self consistently refine max power point
            V=0
            self.Irc=Irc0 # Radiative coupling from upper stack 
            Ij=np.copy(Ijx) # Current absorbed in each junction
            for i in range(topJ,bottomJ-1,-1): # From top to bottom: get photocurrent in each junction including radiative coupling
                Ij[i]+=self.Irc # Include emitted light from upper junction
                if (Ij[i]>Imax): # If there is excess current in this junction, radiative coupling
                    self.Irc=self.beta*self.ERE*(Ij[i]-Imax) #radiative coupling 
                else:
                    self.Irc=0
            Ijmin=Ij[bottomJ:topJ+1].min() # Min current in series connected stack
            I=Ijmin*np.arange(0.75,1,0.0001) # IV curve sampling   
            for i in range(topJ,bottomJ-1,-1): # From top to bottom: Sample IV curve, get I0
                if (i==bottomJ): # The bottom junction of each series connected stack has some additional photon recycling due to partial back reflection of luminescence
                    backLoss=1 # This is the loss due to an air gap between mechanically stacked cells. Same loss assumed for back contact metal mirrors.
                else:
                    backLoss=self.beta
                I0=(1+backLoss) * 2*np.pi*q * np.exp(-1*self.gaps[i]*q/kT) * kT**3*((self.gaps[i]*q/kT+1)**2+1) / (con.h*hc**2) / self.ERE # Dark current at V=0 in A / m2 s
                V+=(kT/q)*np.log((Ij[i]-I)/I0+1) # add voltage of series connected cells
            V-=self.R*I # V drop due to series resistance
            Imax=I[np.argmax(I*V)] # I at max power point
            if int(Imax*1e5)==int(Ijmin*0.75*1e5):
                print ('Currents lower than Ijmin*0.75 need to be sampled to find the max pow point.')
                print ('Change 0.75 to a lower value in code for tandems.serie.')
                pdb.set_trace()
            Imaxs.append(Imax)
        if len(Imaxs)>10:
            print ('Self consistency is slowing convergence while finding the maximum power point.')
            print ('ERE or beta might be too high.')
            print ('Current at the maximum power point is converging as:', Imaxs)
            pdb.set_trace()
        self.Itotal+=Ijmin
        self.Pout+=(I*V).max()
        return
    def stack(self,spec): # Use a single spectrum to get power from 4 terminal tandem, if numTop=junctions the result is for 2 terminal tandem.
        self.Irc=0 # For top cell there is no radiative coupling from upper cell
        self.serie(self.junctions-1,self.junctions-self.numTop,spec) # Add efficiency from top stack, numTop is number of juntions in top stack
        self.serie(self.junctions-self.numTop-1,0,spec) # Add efficiency from bottom stack
        return
    def sample(self): # Sample efficiencies for random band gaps. The multijunction type is defined with junctions,numTop   
        startTime=time.time()
        ncells=0 #number of calculated gap combinations
        nres=0
        effmax=0
        Eshifts=Es[self.junctions] # initial guess at where the eff maxima are. The search space is expanded as needed if high eff are found at the edges of the search space.        
        Escatter=Ef[self.junctions]
        # REMOVE SEED ! unless you want to reuse the same sequence of random numbers (e.g.: compare results after changing one parameter)
        np.random.seed(07022015)
        self.rgaps=np.zeros((self.cells+1000,self.junctions)) # Gaps
        self.auxIs=np.zeros((self.cells+1000,self.junctions)) # Aux array for plotting only
        self.auxeffs=np.zeros((self.cells+1000,self.junctions)) # Aux array for plotting only  
        self.Is=np.zeros((self.cells+1000,10)) # Currents as a function of the number of spectral bins, 0 is standard spectrum 
        self.effs=np.zeros((self.cells+1000,10)) # Efficiencies as a function of the number of spectral bins, 0 is standard spectrum 

        while (nres<self.cells+1000): # loop to randomly sample a large number of gap combinations
            self.gaps=np.zeros(self.junctions+1) #the last gap is not the top junction, it is the high energy limit of the spectrum
            self.gaps[-1]=4.42 
            lastgap=0
            for i in range(0,self.junctions):     # From bottom to top: define random gaps
                self.gaps[i]=lastgap+np.random.rand()*Escatter[i]+Eshifts[i] #define gaps from random differences in energy 
                lastgap=self.gaps[i]  
                              
            self.Itotal=0
            self.Pout=0
            self.stack(1) # calculate power out with average spectrum (1 bin) 
            eff=self.Pout/P[self.d,1]/self.concentration # get efficiency
            if (eff>effmax-self.effmin-0.01): # If gap combination is good, do more work on it
                if int(100000*time.time() % 3000)==1: # Show calculation progress from time to time
                    print ('Tried',ncells,', got ',nres,' candidate gap combinations.')
                for gi,gap in enumerate(self.gaps[:-1]): # Expand edges of search space as needed
                    if gi==0: 
                        gapDiff=self.gaps[0]
                        if Eshifts[0]<0.45: gapDiff=0.46
                    else:
                        gapDiff=self.gaps[gi]-self.gaps[gi-1]
                    if Eshifts[gi]>gapDiff-0.01: Eshifts[gi]=gapDiff-0.01 # Expand edges of search space as needed
                    if Escatter[gi]<(gapDiff-Eshifts[gi]+0.01): Escatter[gi]=(gapDiff-Eshifts[gi]+0.01) # Expand edges of search space as needed
                if (eff>effmax):
                    effmax=eff
                self.rgaps[nres,:]=self.gaps[:-1]
                self.effs[nres,1]=eff 
                self.Is[nres,1]=self.Itotal 
                for i in self.numbins: # loop for number of bins
                    self.Itotal=0
                    Pin=0
                    self.Pout=0                  
                    for j in range(0,a1123456789[i]): # loop for bin index.    a1123456789=[1,1,2,3,4,5,6,7,8,9]
                        self.stack(bindex[i][j]) # calculate power out for each spectra and add to Pout to calculate average efficiency
                        Pin+=P[self.d,bindex[i][j]]
                    self.effs[nres,i]=self.Pout/Pin/self.concentration
                    self.auxeffs[nres,:]=np.zeros(self.junctions)+self.effs[nres,i]
                    self.auxIs[nres,:]=np.zeros(self.junctions)+self.Itotal/a1123456789[i] # a1123456789=[1,1,2,3,4,5,6,7,8,9]
                    self.Is[nres,i]=self.Itotal/a1123456789[i]
                nres+=1                
            ncells+=1
        mask = self.auxeffs > self.auxeffs.max()-self.effmin
        self.rgaps = self.rgaps[mask] # Discard results below the efficiency threshold set by effmin
        self.auxIs = self.auxIs[mask] # As a side effect of this cut off, arrays are flattened
        self.auxeffs = self.auxeffs[mask] # [:,0]
        threshold = self.effs[:,self.numbins[-1]].max()-self.effmin
        mask = self.effs[:,self.numbins[-1]] > threshold
        self.effs = self.effs[mask]
        self.Is = self.Is[mask]
        tiempo=int(time.time()-startTime)
        res=np.size(self.rgaps)/self.junctions
        print ('Calculated ',ncells, ' and saved ', res, ' gap combinations in ',tiempo,' s :',res/(tiempo+1),' results/s')
    def plot(self):
        startTime=time.time()
        print ('I min, I max : ',self.auxIs.min(),self.auxIs.max())
        print ('eff min, eff max : ',self.auxeffs.min(),self.auxeffs.max())
        res=np.size(self.rgaps)/self.junctions
        
        #rgaps=self.rgaps.flatten() 
        #auxeffs=self.auxeffs.flatten()
        #rIs_=(self.auxIs[:,0]-self.auxIs[:,0].min())/(self.auxIs[:,0].max()-self.auxIs[:,0].min())
        Is_=np.copy(self.auxIs)
        Is_=(Is_-Is_.min())/(Is_.max()-Is_.min())
        srIs_=(self.Is[:,self.numbins[-1]]-self.Is[:,self.numbins[-1]].min())/(self.Is[:,self.numbins[-1]].max()-self.Is[:,self.numbins[-1]].min())
        if self.convergence:
            plt.figure()
            plt.xlabel('Number of spectral bins')
            plt.ylabel('Absolute efficiency change \n by increasing number of bins (%)')
            for i in range(0,int(res)):
                diffs=[]
                for j in self.numbins[:-1]:
                    diffs.append(100*(self.effs[i,j+1]-self.effs[i,j]))
                plt.plot(self.numbins[:-1],diffs,color=LGBT(srIs_[i]),linewidth=0.1) #  color=LGBT(auxIs[i*self.junctions])
            plt.savefig('Convergence '+self.name+' '+str(int(self.junctions))+' '+str(int(self.numTop))+' '+str(int(self.concentration))+' '+str(int(startTime)),dpi=600)
            plt.show()
        plt.figure()
        plt.ylim(100*(self.effs[:,self.numbins[-1]].max()-self.effmin),100*self.effs[:,self.numbins[-1]].max()+0.1)
        plt.xlim(0.4,2.6)
        #plt.grid(True)
        plt.minorticks_on()
        plt.tick_params(direction='out',which='minor')
        plt.tick_params(direction='inout',pad=6)
        plt.ylabel('Yearly averaged efficiency (%)')
        plt.xlabel('Energy gaps (eV)')
        plt.plot([0.50,0.50],[0,100],c='grey',linewidth=0.5)
        plt.text(0.5,100*self.auxeffs.max()+.2,'A')
        plt.plot([0.69,0.69],[0,100],c='grey',linewidth=0.5)
        plt.text(0.69,100*self.auxeffs.max()+.2,'B')
        plt.plot([0.92,0.92],[0,100],c='grey',linewidth=0.5)
        plt.text(0.92,100*self.auxeffs.max()+.2,'C')
        plt.plot([1.1,1.1],[0,100],c='grey',linewidth=0.5)
        plt.text(1.1,100*self.auxeffs.max()+.2,'D')
        plt.plot([1.33,1.33],[0,100],c='grey',linewidth=0.5)
        plt.text(1.33,100*self.auxeffs.max()+.2,'E1')
        plt.plot([1.63,1.63],[0,100],c='grey',linewidth=0.5)
        plt.text(1.63,100*self.auxeffs.max()+.2,'E2')
        plt.scatter(self.rgaps,100*self.auxeffs,c=Is_,s=70000/len(self.effs[:,self.numbins[-1]]), edgecolor='none', cmap=LGBT)

        #if not self.convergence:
        plt.savefig(self.name+' '+str(int(self.junctions))+' '+str(int(self.numTop))+' '+str(int(self.concentration))+' '+str(int(startTime)),dpi=600)
        plt.show()        
        plt.xlim(100*(self.auxeffs.max()-self.effmin),100*self.auxeffs.max())
        plt.xlabel('Yearly averaged efficiency (%)')
        plt.ylabel('Count')
        plt.hist(100*self.effs[:,self.numbins[-1]], bins=30)
        #if not self.convergence:
        plt.savefig('Hist Eff '+self.name+' '+str(int(self.junctions))+' '+str(int(self.numTop))+' '+str(int(self.concentration))+' '+str(int(startTime)),dpi=600)      
        plt.show()
    def save(self):
        with open(self.name+' '+str(int(self.junctions))+' '+str(int(self.numTop))+' '+str(int(self.concentration))+' '+str(int(time.time())), "w") as f:
            f.write(json_tricks.dumps(self))
    def load(self,fname):
        with open(fname,"r") as f:
            t0=f.readlines()
            t0=''.join(t0)
            return json_tricks.loads(t0)
            
def generate_spectral_bins():
    # NECESARY CHANGES IN SMARTS 2.9.5 SOURCE CODE
    # Line 189
    #       batch=.TRUE.
    # Line 1514
    #      IF(Zenit.LE.80.)GOTO 13
    #      WRITE(16,103,iostat=Ierr24)Zenit
    # 103  FORMAT(//,'Zenit = ',F6.2,' is > 80 deg. (90 in original code)'
    # This change is needed because trackers are shadowed by neighboring trackers when the sun is near the horizon. 
    # Zenit 80 is already too close to the horizon to use in most cases due to shadowing issues.

    numres=20000 # number of generated spectra

    directos=np.zeros((numres,2002))
    globales=np.zeros((numres,2002))
    EPR650d=np.zeros(numres) # Ivan Garcia's criteria for binning, store EPR value for each spectrum
    EPR650g=np.zeros(numres) # Ivan Garcia's criteria for binning
    Pd=np.zeros(numres) # Power for each spectrum
    Pg=np.zeros(numres)

    os.chdir('/home/jose/SMARTS')
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
            t2+='%.2f' % (120*np.arccos(2*np.random.rand()-1)/np.pi-60)+' '  # 60 > Latitude > -60
            t2+='%.2f' % (360*np.random.rand()-180)+' 0\t\t\t!Card 17a Y M D H Lat Lon Zone\n\n'
            with open('smarts295.inp.txt' , "w") as fil:
                fil.write(t1+t2)
            os.remove("smarts295.ext.txt")
            os.remove("smarts295.out.txt")
            sc+=1
            ou=sub.check_call('./smartsb', shell=True) # execute SMARTS
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
    np.save('scs', Iscs)
