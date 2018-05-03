#!/usr/bin/env python
# Copyright (C) 2017 Jingtao Lai

import os
import numpy as np
from netCDF4 import Dataset
import subprocess as sub

def build_paleo_modifier(delta_T=[0.0], frac_P=[1.0], *args, **kwargs):
    climate_forcing_dir = kwargs['climate_forcing_dir']
    out_dir = kwargs['outdir']
    cmd = ['/usr/bin/ncgen', '-b', '-o',
           os.path.join(climate_forcing_dir, 'paleo_modifier.nc'),
           os.path.join(climate_forcing_dir, 'paleo_modifier.cdl')]
    print ' '.join(cmd)
    sub.call(cmd)
    for P in frac_P:
        for T in delta_T:
            cmd = ['ncap2', '-O', '-s', 'delta_T(0)={};frac_P(0)={}'.format(T, P),
                   os.path.join(climate_forcing_dir, 'paleo_modifier.nc'),
                   os.path.join(out_dir, 'paleo_modifier_T_{}_P_{}.nc'.format(T, P))]
            print ' '.join(cmd)
            sub.call(cmd)

def build_constant_climate(infile=None, outfile=None, *args, **kwargs):
    try:
        air_temp_mean_annual = kwargs['air_temp_mean_annual']
    except:
        air_temp_mean_annual = 9.5-2
    try:
        air_temp_mean_july = kwargs['air_temp_mean_july']
    except:
        air_temp_mean_july = 15.5-2
    try:
        precipitation = kwargs['precipitation']
    except:
        precipitation = 2000.0
    try:
        usurf = kwargs['usurf']
    except:
        usurf = 0.0

    write_data(infile, outfile, air_temp_mean_annual=air_temp_mean_annual,
               air_temp_mean_july=air_temp_mean_july, precipitation=precipitation,
               usurf=usurf)

def write_data(infile=None, outfile=None, *args, **kwargs):
    # dimensions of Olympic
    x_dim = 200
    y_dim = 180
    dim_value = {'x': np.arange(x_dim),
                 'y': np.arange(y_dim)}
    if not(infile is None):
        indata = Dataset(infile, 'r')
        x_dim, = indata.variables['x'].shape
        y_dim, = indata.variables['y'].shape
        dim_value['x'] = indata.variables['x'][:]
        dim_value['y'] = indata.variables['y'][:]
        indata.close()
    outdata = Dataset(outfile, 'w')

    air_temp_mean_annual = kwargs['air_temp_mean_annual']
    air_temp_mean_july = kwargs['air_temp_mean_july']
    precipitation = kwargs['precipitation']
    usurf = kwargs['usurf']

    dim_len = {'x': x_dim, 'y': y_dim, 'time': None}
    for dim_name in ['x', 'y', 'time']:
        try:
            outdata.createDimension(dim_name, dim_len[dim_name])
            outdata.createVariable(dim_name, np.float64, (dim_name,))
            if not(dim_len[dim_name] is None):
                outdata.variables[dim_name][:] = dim_value[dim_name]
        except:
            continue

    var_name_list = ['air_temp_mean_annual',
                     'air_temp_mean_july',
                     'precipitation',
                     'usurf']
    var_value = {'air_temp_mean_annual': air_temp_mean_annual,
                 'air_temp_mean_july': air_temp_mean_july,
                 'precipitation': precipitation,
                 'usurf': usurf}
    var_unit = {'air_temp_mean_annual': 'celsius',
                 'air_temp_mean_july': 'celsius',
                 'precipitation': 'kg m-2 year-1',
                 'usurf': 'm'}
    for var_name in var_name_list:
        try:
            var = outdata.variables[var_name]
        except:
            var = outdata.createVariable(var_name, np.float64, ('y', 'x',))
        var[:] = var_value[var_name]
        var.units = var_unit[var_name]

    outdata.variables['usurf'].standard_name = 'surface_altitude'

    outdata.close()

if __name__=='__main__':
    infile = '../bed_dem/test_dem.nc'
    outfile = 'test_climate.nc'
    build_constant_climate(infile=infile, outfile=outfile)
    build_paleo_modifier(delta_T = [100, 200], frac_P=[-100, -200])
