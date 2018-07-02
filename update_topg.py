#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Copyright (C) 2018 Jingtao Lai
"""

import numpy as np
from argparse import ArgumentParser
from netCDF4 import Dataset
import sys
try:
    import subprocess32 as sub
except:
    import subprocess as sub

def update_topg(state_file=None, spatial_file=None, outfile=None, k_g=1e-4, dt=100):
    #cmd = ['ncks', '-v', 'thk,topg', '-O', state_file, outfile]
    cmd = ['cp', state_file, outfile]
    sub.call(cmd)

    spatial_data = Dataset(spatial_file, 'r')
    u_s_mean = np.mean(spatial_data.variables['velbase_mag'][:], axis=0)
    spatial_data.close()

    outdata = Dataset(outfile, 'a')
    outdata.variables['topg'][:] -= k_g*u_s_mean*dt
    outdata.close()

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--state_file', dest='state_file', default=None)
    parser.add_argument('--spatial_file', dest='spatial_file', default=None)
    parser.add_argument('--outfile', dest='outfile', default=None)
    parser.add_argument('--k_g', dest='k_g', type=float, default=1e-4)
    parser.add_argument('--dt', dest='dt', type=float, default=100)

    options = parser.parse_args()
    update_topg(state_file=options.state_file,
                spatial_file=options.spatial_file,
                outfile=options.outfile,
                k_g=options.k_g, dt=options.dt)
