Calculate yearly average efficiencies for multijunction tandem solar cells

Random sampling of multijunction photovoltaic efficiencies. Jose M. Ripalda

The main file is tandems.py

Tested with Python 2.7 and 3.6

SMARTS 2.9.5 is required only to generate a new set of random spectra. 

File "data/lat40.npy" can be used instead of SMARTS to load a set of averaged spectra.

Clone or download from https://github.com/Ripalda/Tandems to obtain full set of spectra (about 600 MB).

genBins.py imports tandems.py to generate sets of proxy spectra 

Project Organization
--------------------

    .
    ├── examples
    ├── FARMS-NIT-clustered-spectra-USA
    ├── maps
    ├── tandems
    │   └── data
    └── tests


INSTALL
==============================
Install from pypi
```bash
pip install tandems
```

Development install

```bash
git clone https://github.com/ripalda/tandems
cd Tandems/ 
pip install - e .
```

USAGE EXAMPLE
==============================


```python
import tandems

tandems.docs()

eff = tandems.effs(junctions=4, bins=6, concentration=500)    #    Include as many or as few options as needed.
eff.findGaps()
eff.plot() # Figures saved to PNG files.

eff.save() # Data saved for later reuse/replotting. Path and file name set in eff.name, some parameters and timestamp are appended to filename

eff2 = tandems.deepcopy(eff)
eff2.__init__(junctions=4,bins=8, concentration=1, R=4e-5)  # Change input parameters but keep previously found set of optimal gap combinations.
eff2.recalculate() # Recalculate efficiencies for previously found set of optimal gap combinations.
eff2.compare(eff) # Compares efficiencies in two datasets by doing eff2 - eff. Plots difference and saves PNG files.

# eff = tandems.load('/path/and file name here') # Load previusly saved data
# eff.results()
# eff.plot()

# The .npy files with the spectra used to calculate the yearly average efficiency have been generated with genBins.py
```
