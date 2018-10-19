
Calculate yearly average efficiencies for multijunction tandem solar cells

Random sampling of multijunction photovoltaic efficiencies. Jose M. Ripalda

The main file is tandems.py

genBins.py imports tandems.py to generate sets of proxy spectra 

Import tandems from the current path or
move all files to your standard location for python modules.
This would be something like ~/.local/lib/python3.6/site-packages/

<!--Requires doing "pip install json_tricks" before running-->

Tested with Python 2.7 and 3.6

SMARTS 2.9.5 is required only to generate a new set of random spectra. 

File "lat40.npy" can be used instead of SMARTS to load a set of averaged spectra.

Project Organization
--------------------

    .
    ├── AUTHORS.md
    ├── LICENSE
    ├── README.md
    ├── docs
    ├── notebooks
    │   └── figures
    └── data

USAGE EXAMPLE
==============================


```python
import tandems

tandems.docs()

eff = tandems.effs(junctions=4, bins=6, concentration=500)    #    Include as many or as few options as needed.
eff.findGaps()
eff.plot() # Figures saved to PNG files.

eff.save() # Data saved for later reuse/replotting. Path and file name set in eff.name, some parameters and timestamp are appended to filename

eff2 = tandems.copy.deepcopy(eff)
eff2.__init__(junctions=4,bins=8, concentration=1, R=4e-5)  # Change input parameters but keep previously found set of optimal gap combinations.
eff2.recalculate() # Recalculate efficiencies for previously found set of optimal gap combinations.
eff2.compare(eff) # Compares efficiencies in two datasets by doing eff2 - eff. Plots difference and saves PNG files.

eff = tandems.load('/path/and file name here') # Load previusly saved data
eff.results()
eff.plot()

# The .npy files with the spectra used to calculate the yearly average efficiency have been generated with genBins.py
```

