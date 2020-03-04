#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 16:38:17 2020

@author: jose
"""

import tandems

tandems.generate_spectral_bins(fname='40.85_-115.82_1683_-8_axis', loadFullSpectra=True, glo_dir=[0])

s = tandems.effs(cells = 140, junctions=3, concentration=1, R=4e-5, cloudCover=0, bins=list(range(1,31)), convergence=True, T_from_spec_file=True, expected_eff=0.3478, specsFile='40.85_-115.82_1683_-8_axis.clusters.npz')
s.findGaps()
s.plot()