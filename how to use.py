# Random sampling of multijunction photovoltaic efficiencies. Jose M. Ripalda
# Requires doing "pip install json_tricks" before running
# Tested with Python 2.7 and 3.6
# SMARTS 2.9.5 is required only to generate a new set of random spectra. 
# File "scs2.npy" can be used instead of SMARTS to load a set of binned spectra.

import tandems

help(tandems.effis)

effi=tandems.effis(junctions=4,bins=6) # The list of options that can be used here can be found by typing "help(tandems.effis)"
effi.sample()
effi.plot()

# generated figures are saved to the working directory

