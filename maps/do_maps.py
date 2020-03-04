#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 09:25:55 2019

Efficiency maps using bins from NSRDB

@author: jose

"""
__author__ = 'Jose M. Ripalda'
__version__ = 0.02

import os
import tandems
tandems.Dpath=''
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from scipy.interpolate import griddata

print('V', __version__)

ndatos = 913

lats = np.zeros(ndatos)
lons = np.zeros(ndatos)
fns0s = np.zeros(ndatos, dtype=int)
latsb = np.zeros(ndatos)
lonsb = np.zeros(ndatos)

tilt_power = np.zeros(ndatos)
axis_power = np.zeros(ndatos)


dSift = np.zeros(ndatos)
dSifts = np.zeros(ndatos)
dSifte = np.zeros(ndatos)
dSiftes = np.zeros(ndatos)


dCdTeft = np.zeros(ndatos)
dCdTefts = np.zeros(ndatos)
dCdTefte = np.zeros(ndatos)

dP = np.zeros(ndatos)


dSiha = np.zeros(ndatos)
dSihas = np.zeros(ndatos)
dSihae = np.zeros(ndatos)
dSihaes = np.zeros(ndatos)

d2jha = np.zeros(ndatos)
d3jha = np.zeros(ndatos)

d2jhas = np.zeros(ndatos)
d3jhas = np.zeros(ndatos)

d2jhae = np.zeros(ndatos)
d3jhae = np.zeros(ndatos)

d6j = np.zeros(ndatos)
d6js = np.zeros(ndatos)


d2j3t = np.zeros(ndatos)
d3j3t = np.zeros(ndatos)
d3j3ts = np.zeros(ndatos)
d2j3ts= np.zeros(ndatos)
dgaas = np.zeros(ndatos)
dgainp = np.zeros(ndatos)
dgaas3t = np.zeros(ndatos)
dgainp3t = np.zeros(ndatos)


def mapa(d, nombre='mapa',lats=lats, lons=lons, eje='Improvement factor' ):
    print(nombre)

    # define INTERPOLATION grid
    xi = np.linspace(lons.min()-1,lons.max()+1,150)
    yi = np.linspace(lats.min()-1,lats.max()+1,100)

    # grid the data.
    lons = np.concatenate((lons,np.array([-100,-100,-100000,100000, -112, -98.18])))
    lats = np.concatenate((lats,np.array([-100000,100000,40,40, 31, 25])))
    d = np.concatenate((d,np.array([d.mean(),d.mean(),d.mean(),d.mean(), d[np.where(lats == 32.17)], d[np.where(lats == 26.29)]])))
    zi = griddata((lons, lats), d, (xi[None,:], yi[:,None]), method='linear')

    fig = plt.figure()
    fig.add_axes([0.1,0.1,0.8,0.8])
    m = Basemap(projection='stere',lon_0=-100,lat_0=20.,lat_ts=20,\
                llcrnrlat=22,urcrnrlat=49,\
                llcrnrlon=-123,urcrnrlon=-62,\
                rsphere=6371200.,resolution='l',area_thresh=10000)
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()
    parallels = np.arange(0.,90,10.)
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
    meridians = np.arange(180.,360.,10.)
    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
    #m.bluemarble()
    #m.fillcontinents(color='lightgray',zorder=0)

    xm, ym = np.meshgrid(xi, yi)
    xmm, ymm = m(xm, ym)

    #zi = maskoceans(xm, ym, zi)
    # m.contour(xm,ym,sampley,200,cmap=tandems.LGBT)
    #colors = [(0.3, 0, 0.2), (0.7, 0, 0.5), (0, 0, 1), (0, 0.5, 0.5), (0, 1, 1), (0, 1, 0), (0.7, 0.9, 0), (1, 0.9, 0), (1, 0.5, 0.2), (1, 0, 0), (1, 0.6, 0.8)]  # B -> G -> R
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
    m.pcolormesh(xmm,ymm,zi,cmap=LGBT)

    # m.scatter(lons, lats, c=d, cmap=tandems.LGBT, latlon=True, zorder=10)
    m.colorbar(location='bottom',pad="10%", label=eje)
    plt.savefig(nombre, dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure()
    plt.xlabel(eje)
    plt.ylabel('Count.npy')
    dpd = pd.Series(d)
    plt.hist(dpd.dropna(), bins=30)
    #plt.savefig(nombre + ' Hist ', dpi=300, bbox_inches="tight")
    plt.show()



list_main = os.listdir('FARMS-NP')

f = 0
fc = 0

filetype = 'tilt.clusters.npz'

nf = 0

for filename in list_main:
    fns = filename.split('_')

    if fns[-1] == filetype and float(fns[0]) > 23:

        if not os.path.isfile('FARMS-NP/' + filename.replace('tilt','axis')):
            print('FARMS-NP/' + filename.replace('tilt','axis'), os.path.isfile('FARMS-NP/' + filename.replace('tilt','axis')))
        else:
            nf += 1

        print(filename)
        lats[fc] = fns[0]
        lons[fc] = fns[1]
        filename = 'FARMS-NP/' + filename

        s = tandems.effs(T_from_spec_file=True, junctions=1,cells=1,concentration=1, R=4e-5, cloudCover=0, EQE=1, gaps=[1.126], bins=[18], expected_eff = 0.2587, specsFile=filename)
        s.useBins( s.bins, 0 )
        dSift[fc] = s.auxEffs.max()
        dSifte[fc] = s.kWh(s.auxEffs.max())
        tilt_power[fc] = s.P[0, 1]


        s = tandems.effs(T_from_spec_file=True, junctions=1,cells=1,concentration=1, R=4e-5, cloudCover=0, EQE=1, gaps=[1.126], bins=[18], expected_eff = 0.2587, specsFile=filename)
        s.useBins( s.bins, 0 )
        dSift[fc] = s.auxEffs.max()

        s = tandems.effs(T_from_spec_file=True, junctions=1,cells=1,concentration=1, R=4e-5, cloudCover=0, EQE=1, gaps=[1.126], bins=[-18], expected_eff = 0.2587, specsFile=filename)
        s.useBins( s.bins, 0 )
        dSifts[fc] = s.auxEffs.max()
        dSiftes[fc] = s.kWh(s.auxEffs.max())


        s = tandems.effs(T_from_spec_file=True, junctions=1, cells=1, concentration=1, R=4e-5, cloudCover=0, gaps=[1.45], bins=[18], expected_eff = 0.226, ERE=1e-4, specsFile=filename)
        s.useBins( s.bins, 0 )
        dCdTeft[fc] = s.auxEffs.max()

        s = tandems.effs(T_from_spec_file=True, junctions=1, cells=1, concentration=1, R=4e-5, cloudCover=0, gaps=[1.45], bins=[-18], expected_eff = 0.226, ERE=1e-4, specsFile=filename)
        s.useBins( s.bins, 0 )
        dCdTefts[fc] = s.auxEffs.max()

        s = tandems.effs(T_from_spec_file=True, junctions=1, cells=1, concentration=1, R=4e-5, cloudCover=0, gaps=[1.5], bins=[18], expected_eff = 0.24, specsFile=filename)
        s.useBins( s.bins, 0 )
        dP[fc] = s.auxEffs.max()




        s = tandems.effs(T_from_spec_file=True, junctions=1,cells=1,concentration=1, R=4e-5, cloudCover=0, EQE=1, gaps=[1.126], bins=[18], expected_eff = 0.2587, specsFile=filename.replace('tilt','axis'))
        s.useBins( s.bins, 0 )
        dSiha[fc] = s.auxEffs.max()
        dSihae[fc] = s.kWh(s.auxEffs.max())
        axis_power[fc] = s.P[0, 1]

        s = tandems.effs(T_from_spec_file=True, junctions=1,cells=1,concentration=1, R=4e-5, cloudCover=0, EQE=1, gaps=[1.126], bins=[-18], expected_eff = 0.2587, specsFile=filename.replace('tilt','axis'))
        s.useBins( s.bins, 0 )
        dSihas[fc] = s.auxEffs.max()
        dSihaes[fc] = s.kWh(s.auxEffs.max())

        s = tandems.effs(T_from_spec_file=True, junctions=2,cells=1,concentration=1, R=4e-5, cloudCover=0, gaps=[ 1.126,  1.687], bins=[18], expected_eff = 0.3181, specsFile=filename.replace('tilt','axis'))
        s.useBins( s.bins, 0 )
        d2jha[fc] = s.auxEffs.max()
        d2jhae[fc] = s.kWh(s.auxEffs.max())

        s = tandems.effs(T_from_spec_file=True, junctions=3, cells=1,concentration=1, R=4e-5, cloudCover=0, gaps=[ 1.126, 1.48, 1.94], bins=[18], expected_eff = 0.3478, specsFile=filename.replace('tilt','axis'))
        s.useBins( s.bins, 0 )
        d3jha[fc] = s.auxEffs.max()
        d3jhae[fc] = s.kWh(s.auxEffs.max())

        s = tandems.effs(T_from_spec_file=True, junctions=6, cells=1, concentration=1, R=4e-5, cloudCover=0, gaps=[0.7, 0.97, 1.19, 1.45, 1.76, 2.19], bins=[18], expected_eff = 0.4, specsFile=filename.replace('tilt','axis'))
        s.useBins( s.bins, 0 )
        d6j[fc] = s.auxEffs.max()

        s = tandems.effs(T_from_spec_file=True, junctions=6, cells=1, concentration=1, R=4e-5, cloudCover=0, gaps=[0.7, 0.97, 1.19, 1.45, 1.76, 2.19], bins=[-18], expected_eff = 0.4, specsFile=filename.replace('tilt','axis'))
        s.useBins( s.bins, 0 )
        d6js[fc] = s.auxEffs.max()



        s = tandems.effs(T_from_spec_file=True, junctions=2, topJunctions=1, cells=1,concentration=1, R=4e-5, cloudCover=0, gaps=[ 1.126,  1.687], bins=[18], expected_eff = 0.3181, specsFile=filename.replace('tilt','axis'))
        s.useBins( s.bins, 0 )
        d2j3t[fc] = s.auxEffs.max()

        s = tandems.effs(T_from_spec_file=True, junctions=2, topJunctions=1, cells=1,concentration=1, R=4e-5, cloudCover=0, gaps=[ 1.126,  1.687], bins=[-18], expected_eff = 0.3181, specsFile=filename.replace('tilt','axis'))
        s.useBins( s.bins, 0 )
        d2j3ts[fc] = s.auxEffs.max()

        s = tandems.effs(T_from_spec_file=True, junctions=3, topJunctions=2, cells=1,concentration=1, R=4e-5, cloudCover=0, gaps=[ 1.126, 1.48, 1.94], bins=[18], expected_eff = 0.3478, specsFile=filename.replace('tilt','axis'))
        s.useBins( s.bins, 0 )
        d3j3t[fc] = s.auxEffs.max()

        s = tandems.effs(T_from_spec_file=True, junctions=3, topJunctions=2, cells=1,concentration=1, R=4e-5, cloudCover=0, gaps=[ 1.126, 1.48, 1.94], bins=[-18], expected_eff = 0.3478, specsFile=filename.replace('tilt','axis'))
        s.useBins( s.bins, 0 )
        d3j3ts[fc] = s.auxEffs.max()

        s = tandems.effs(T_from_spec_file=True, junctions=2, cells=1,concentration=1, R=4e-5, cloudCover=0, gaps=[ 1.126,  1.42], bins=[18], expected_eff = 0.3181, specsFile=filename.replace('tilt','axis'))
        s.useBins( s.bins, 0 )
        dgaas[fc] = s.auxEffs.max()

        s = tandems.effs(T_from_spec_file=True, junctions=2, cells=1,concentration=1, R=4e-5, cloudCover=0, gaps=[ 1.126,  1.85], bins=[18], expected_eff = 0.3181, specsFile=filename.replace('tilt','axis'))
        s.useBins( s.bins, 0 )
        dgainp[fc] = s.auxEffs.max()

        s = tandems.effs(T_from_spec_file=True, junctions=2, topJunctions=1, cells=1,concentration=1, R=4e-5, cloudCover=0, gaps=[ 1.126,  1.42], bins=[18], expected_eff = 0.3181, specsFile=filename.replace('tilt','axis'))
        s.useBins( s.bins, 0 )
        dgaas3t[fc] = s.auxEffs.max()

        s = tandems.effs(T_from_spec_file=True, junctions=2, topJunctions=1, cells=1,concentration=1, R=4e-5, cloudCover=0, gaps=[ 1.126,  1.85], bins=[18], expected_eff = 0.3181, specsFile=filename.replace('tilt','axis'))
        s.useBins( s.bins, 0 )
        dgainp3t[fc] = s.auxEffs.max()

        fc += 1
        print('Location',fc)


np.save('lats2020m', lats)
np.save('lons2020m', lons)

np.save('tilt_power', tilt_power)
np.save('axis_power', axis_power)


np.save('dSift', dSift)
np.save('dSiha', dSiha)
np.save('d2jha', d2jha)
np.save('d3jha', d3jha)

np.save('dSifts', dSifts)
np.save('dSihas', dSihas)
np.save('dSiftes', dSiftes)
np.save('dSihaes', dSihaes)
np.save('d2jhas', d2jhas)
np.save('d3jhas', d3jhas)


np.save('dSifte', dSifte)
np.save('dSihae', dSihae)
np.save('d2jhae', d2jhae)
np.save('d3jhae', d3jhae)

np.save('dCdTeft', dCdTeft)
np.save('dCdTefts', dCdTefts)
np.save('dCdTefte', dCdTefte)
np.save('dP', dP)

np.save('d6jft', d6j)
np.save('d6jfts', d6js)

np.save('d2j3t', d2j3t)
np.save('d3j3t', d3j3t)
np.save('d2j3ts', d2j3ts)
np.save('d3j3ts', d3j3ts)

np.save('dgaas', dgaas)
np.save('dgainp', dgainp)
np.save('dgaas3t', dgaas3t)
np.save('dgainp3t', dgainp3t)


s = tandems.effs(Tmin=25 + 273.15, deltaT=np.array([0, 0]), junctions=1,cells=1,concentration=1, R=4e-5, cloudCover=0, EQE=1, gaps=[1.126], bins=[0], expected_eff = 0.2587, specsFile='FARMS-NP/31.05_-93.18_94_-6_tilt.clusters.npz')
s.useBins( s.bins, 0 )
print(s.auxEffs.max(), s.kWh(s.auxEffs.max()), s.P[0,0])


eta0 = s.auxEffs.max()  # 0.2707 %
yield0 = s.kWh(s.auxEffs.max())  # 526.55 kWh / m2 / year
rating = eta0 * 1000

mapa(dSift/eta0, 'mapa rel eff ft',lats=lats, lons=lons, eje='Yearly average efficiency / Standard efficiency')

mapa(0.9 * 1000 * dSifte / rating, 'mapa specific yield ft',lats=lats, lons=lons, eje='Yearly energy yield  / Rated peak power $\mathregular{(kWh \ kW^{-1})}$')


mapa(100*(dSift/dSifts - 1), 'spec fact ft',lats=lats, lons=lons, eje='Spectral correction to the energy yield (%)')


mapa(100*((dP / dSift) - 1), 'p',lats=lats, lons=lons, eje='Perovskite / Si energy yield difference (%)')

mapa(100*((dCdTeft / dCdTefts) - 1), 'cdte spectral', lats=lats, lons=lons, eje='CdTe spectral correction to the energy yield (%)')

mapa(100*(dSihae/dSifte-1), 'mapa e ratio tracking',lats=lats, lons=lons, eje='Yield improvement with tracking (%)')

mapa(100*(dSiha/dSift-1), 'mapa eff ratio tracking',lats=lats, lons=lons, eje='Efficiency improvement with tracking (%)')

mapa(100*(dSiha/dSihas - 1), 'spec fact ha',lats=lats, lons=lons, eje='Spectral correction to the energy yield (%)')

mapa(100*((dSiha/dSihas) / (dSift/dSifts) - 1), 'spec fact ha-ft',lats=lats, lons=lons, eje='HSAT / FT spectral correction ratio (%)')

mapa(0.9 * 1000 * dSihae / rating, 'mapa specific yield ha',lats=lats, lons=lons, eje='Yearly energy yield  / Rated peak power $\mathregular{(kWh \ kW^{-1})}$')

# TODO recording daytimeFraction is needed to report POA irradiance correctly. See tandems.effs.results()
# mapa(axis_power, 'axis power', lats=lats, lons=lons, eje='POA Irradiance on 1-axis tracker')

mapa(100*(d2jha/dSiha - 1), 'mapa eff ratio 2 1',lats=lats, lons=lons, eje='2 junct. / Si   Energy yield advantage (%)')
mapa(100*(d3jha/d2jha - 1), 'mapa eff ratio 3 2',lats=lats, lons=lons, eje='3 junct. / 2 junct.   Energy yield advantage (%)')


mapa(100*(d6j/d6js - 1), 'spec fact 6j',lats=lats, lons=lons, eje='Spectral correction to the energy yield (%)')

mapa(dCdTeft*100, 'mapa eff cdte ft',lats=lats, lons=lons, eje='Efficiency (%)')

mapa(dCdTeft/dSift, 'mapa eff ratio cdte si',lats=lats, lons=lons, eje='Efficiency ratio CdTe / Si')

mapa(100*((dgaas3t / dgaas) - 1), 'gaas 3t',lats=lats, lons=lons, eje='Energy yield improvement with 3 Terminals (%)')

mapa(100*((dgainp3t / dgainp) - 1), 'gainp 3t',lats=lats, lons=lons, eje='Energy yield improvement with 3 Terminals (%)')


mapa(100*((d2j3t / d2j3ts) - 1), 'spec corr 2j 3t',lats=lats, lons=lons, eje='2J 3T spectral correction to the energy yield (%)')
mapa(100*((d3j3t / d3j3ts) - 1), 'spec corr 3j 3t',lats=lats, lons=lons, eje='3J 3T spectral correction to the energy yield (%)')
mapa(dgaas*100, 'mapa gaas on si eff',lats=lats, lons=lons, eje='Efficiency (%)')
mapa(dgainp*100, 'mapa gainp on si eff',lats=lats, lons=lons, eje='Efficiency (%)')
mapa(dgaas3t*100, 'mapa gaas on si 3t eff',lats=lats, lons=lons, eje='Efficiency (%)')
mapa(dgainp3t*100, 'mapa gainp on si 3t eff',lats=lats, lons=lons, eje='Efficiency (%)')




