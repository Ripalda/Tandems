#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:41:15 2020

@author: jose

Converts NREL NSRDB spectral on demand (FARMS-NIT) ziped csv files to numpy zipped arrays (.npz)

"""

import zipfile

import tandems

import pvlib

import glob

from stat import S_ISREG, ST_MTIME, ST_MODE
import os, time


lats = tandems.np.load('lats2020.npy')
lons = tandems.np.load('lons2020.npy')



lats2 = tandems.np.zeros(1999)
lons2 = tandems.np.zeros(1999)

#exists = tandems.np.zeros(1000)

latlon = []

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


for i, lat in enumerate(lats):
    latlon.append(str(lat) + ' ' + str(lons[i]))


def heat_eff(expected_eff=0.3, R=0.05, eff0=0.165):
    """Irradiance modifier to account for the fact that the Sandia cell
    temperature model does not include the effect of efficiency
    """
    return (1 - R - expected_eff) / (1 - R - eff0)


# path to the directory (relative or absolute)
dirpath = 'FARMS-NIT'
#dirpath = 'FARMS-ZIP-VIEJOS'
# get all entries in the directory w/ stats
entries = (os.path.join(dirpath, fn) for fn in os.listdir(dirpath))
entries = ((os.stat(path), path) for path in entries)

# leave only regular files, insert creation date
entries = ((stat[ST_MTIME], path)
           for stat, path in entries if S_ISREG(stat[ST_MODE]))
ind = 0
for mdate, filename in sorted(entries, reverse=True)[:]:
    print(time.ctime(mdate), os.path.basename(filename))

    fs = os.path.basename(filename).split('_')
    if is_number(fs[0]):
        lat = fs[0]
        lon = fs[1]

        lats2[ind] = lat
        lons2[ind] = lon

        substri = 'FARMS-NP/' + str(lat) + '_' + str(lon) + '*' + fs[4].split('.')[0] + '.clusters.npz'
        matches = glob.glob(substri)
        print('matches', substri, len(matches))
        if len(matches) > 0:
            ind += 1
            continue

    with zipfile.ZipFile(dirpath + "/" + os.path.basename(filename), 'r') as zip_ref:
        zip_ref.extractall('FARMS-UNZIP')

    unzipped = os.listdir('FARMS-UNZIP')[0]

    f = open("FARMS-UNZIP/" + unzipped, "r")
    a = f.readline()
    a = f.readline()
    a = a.split(',')
    lat = a[5]
    lon = a[6]
    elevation = a[8]
    timez = a[9]
    f.close()
    panel = unzipped[-8:-4]
    lats2[ind] = lat
    lons2[ind] = lon

    ind += 1
    print(ind, lat, lon)

 #   try:
  #      exists[latlon.index(str(lat) + ' ' + str(lon))] = 1
   # except Exception as e:
    #    exists[len(latlon)] = 1
     #   latlon.append(str(lat) + ' ' + str(lon))

    np_arr = tandems.np.nan_to_num(tandems.np.loadtxt('FARMS-UNZIP/' + unzipped, skiprows=3, delimiter=','))  # Global Horizontal
    i_spectra = tandems.np.zeros((2,8760,len(tandems.wavel))) # global/direct, Time, wavel
    i_spectra[0, :, :] = np_arr[:, 25:] / 1000  # In version 3.0.1 of the FARMS-NIT model, spectral irradiance is in W/m^2/micrometer, need to convert to W/m^2/nm
    irradiance = tandems.np.trapz(i_spectra[0, :, :], x=tandems.wavel, axis=1)
    surface_tilt = np_arr[:, 23]
    surface_azimuth = np_arr[:, 24]
    sol_zenith = np_arr[:, 16]
    sol_azimuth = np_arr[:, 22]
    aoi = tandems.np.nan_to_num(pvlib.irradiance.aoi(surface_tilt, surface_azimuth, sol_zenith, sol_azimuth))
    aim = tandems.np.nan_to_num(pvlib.pvsystem.physicaliam(aoi)) # Angle of incidence modifier
    cell_temp = pvlib.pvsystem.sapm_celltemp(heat_eff() * aim * irradiance, np_arr[:, 21], np_arr[:, 5], model='open_rack_cell_polymerback')['temp_cell']
    i_spectra[0, :, 1] = 1e-6 * np_arr[:, 5] # Temp - These two numbers are made small so they do not have an effect on clustering or binning
    i_spectra[0, :, 2] = 1e-6 * np_arr[:, 21] # Wind - Clustering averages these wind speeds and temps
    i_spectra[0, :, 3] = 1e-1 * tandems.np.nan_to_num(cell_temp) # This number is made larger so it does have an effect on clustering
    i_spectra[0, :, 4] = 1e-3 * irradiance # This number is made larger so it does have an effect on clustering. Vectors are area normalized before clustering (divided by the integral in lambda).
    i_spectra[0, :, 5] = 1e-6 * aoi

    fnm2= 'FARMS-NP/' + str(lat) + '_' + str(lon) + '_' + str(elevation) + '_' + str(timez) + '_' + panel

    os.rename("FARMS-NIT/" + os.path.basename(filename), "FARMS-NIT/" + str(lat) + '_' + str(lon) + '_' + str(elevation) + '_' + str(timez) + '_' + panel + '.zip')

    haynpz = os.path.isfile("FARMS-NP/" + str(lat) + '_' + str(lon) + '_' + str(elevation) + '_' + str(timez) + '_' + panel + '.clusters.npz')

    print('fnm2, haynpz', fnm2, haynpz)

    if not haynpz:

        i_spectra[-1,-1,-1] = 8760
        for i in range(9, 29):
            i_spectra[1, 0, i] = i # This is to have some data for direct spectra, as tandems tandems.generate_spectral_bins expects two sets of spectra (direct and global)
        # print('i_spectra[0, :, :6].mean(axis=0)', i_spectra[0, :, :6].mean(axis=0))
        tandems.np.savez_compressed(fnm2 + '.full', i_spectra)

        tandems.generate_spectral_bins(loadFullSpectra=True, fname=fnm2, bins=[1,18], glo_dir=[0])
        os.remove(fnm2 + '.full.npz')

    os.remove("FARMS-UNZIP/" + unzipped)

tandems.plt.figure()
tandems.plt.xlim(-130,-60)
tandems.plt.ylim(20,50)
tandems.plt.scatter(lons2, lats2, s=2)
tandems.plt.scatter(lons, lats, s=1)

tandems.plt.savefig('latslonsnit2',dpi=300, bbox_inches="tight")











