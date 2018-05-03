#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Copyright (C) 2018 Jingtao Lai
"""

import os
import sys
from collections import OrderedDict
try:
    import subprocess32 as sub
except:
    import subprocess as sub
from argparse import ArgumentParser
import numpy as np
from netCDF4 import Dataset

sys.path.append(os.environ['PARAMPARSER_PATH'])
from paramparser import ParamParser
from build_climate_file import build_constant_climate


def generate_pism_cmd(*args, **kwargs):
    params_file = kwargs['params_file']
    params = ParamParser(params_file)

    pism_prefix = 'pismr'

    system = params.read('system')
    if system in ['keeling']:
        pism_data_dir = os.environ['PISM_DATA_DIR']
    else:
        pism_data_dir = './'
    outdir = params.read('outdir')
    outdir = os.path.join(pism_data_dir, outdir)
    state_dir = 'state'
    scalar_dir = 'scalar'
    spatial_dir = 'spatial'
    postproc_dir = 'postproc'
    initial_dir = 'initial'
    tmp_dir = 'tmp'
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    for subdir in [scalar_dir, spatial_dir, state_dir, postproc_dir, initial_dir, tmp_dir]:
        if not os.path.isdir(os.path.join(outdir, subdir)):
            os.mkdir(os.path.join(outdir, subdir))

    general_params_dict = OrderedDict()
    start_file = params.read('start_file', 'str')
    general_params_dict['i'] = start_file
    bootstrap = params.read('bootstrap', 'bool')
    if bootstrap:
        general_params_dict['bootstrap'] = ''
    start_year = params.read('start_year', 'float')
    end_year = params.read('end_year', 'float')
    outfile = params.read('outfile', 'str')
    general_params_dict['ys'] = start_year
    general_params_dict['ye'] = end_year
    general_params_dict['o'] = outfile
    general_params_dict['o_format'] = params.read('o_format', 'str')
    general_params_dict['o_size'] = params.read('o_size', 'str')
    general_params_dict['options_left'] = ''

    grid_params_dict = generate_grid_description(start_file, bootstrap)

    # stress balance
    stress_balance_params_dict = OrderedDict()
    stress_balance_params_dict['ssa_e'] = params.read('ssa_e', 'float')
    stress_balance_params_dict['pseudo_plastic'] = ''
    stress_balance_params_dict['pseudo_plastic_q'] = params.read('pseudo_plastic_q', 'float')
    stress_balance_params_dict['till_effective_fraction_overburden'] = params.read('till_effective_fraction_overburden', 'float')
    stress_balance_params_dict['plastic_phi'] = params.read('plastic_phi', 'float')
    stress_balance_params_dict['ssafd_ksp_divtol'] = 1e300
    stress_balance_params_dict['cfbc'] = ''
    stress_balance_params_dict['ssa_method'] = 'fd'
    #stress_balance_params_dict['bed_smoother_range'] = 50
    stress_balance_params_dict['sia_flow_law'] = 'gpbld3'

    # climate
    air_temp_mean_annual = params.read('air_temp_mean_annual', 'float')
    air_temp_mean_july = air_temp_mean_annual+5
    precipitation = params.read('precipitation', 'float')
    climate_file = os.path.join(outdir, initial_dir, 'climate_'+outfile)
    #atmosphere_paleo_file = os.path.join(outdir, initial_dir, 'paleo_modifier_'+outfile)
    build_constant_climate(infile=start_file,
                           outfile=climate_file,
                           air_temp_mean_annual=air_temp_mean_annual,
                           air_temp_mean_july=air_temp_mean_july,
                           precipitation=precipitation)
    """
    build_paleo_modifier(delta_T=[0.0],
                         frac_P=[1.0],
                         climate_forcing_dir=os.path.join(pism_work_dir, 'data_sets/climate_forcing'),
                         outdir=os.path.join(outdir, initdata_dir))
    """
    temp_lapse_rate = params.read('temp_lapse_rate', 'float')
    ice_density = 910.
    climate_params_dict = OrderedDict()
    climate_params_dict['atmosphere'] = 'yearly_cycle,lapse_rate'
    climate_params_dict['surface.pdd.factor_ice'] = 4.59 / ice_density  # Shea et al (2009)
    climate_params_dict['surface.pdd.factor_snow'] = 3.04 / ice_density  # Shea et al (2009)
    climate_params_dict['surface.pdd.refreeze'] = 0
    climate_params_dict['atmosphere_yearly_cycle_file'] = climate_file,
    climate_params_dict['atmosphere_lapse_rate_file'] = climate_file,
    climate_params_dict['temp_lapse_rate'] = temp_lapse_rate,
    #climate_params_dict['atmosphere_delta_T_file'] = atmosphere_paleo_file,
    #climate_params_dict['atmosphere_frac_P_file'] = atmosphere_paleo_file

    # hydrology diffuse
    hydro_params_dict = OrderedDict()
    hydro_params_dict['hydrology'] = 'null'
    hydro_params_dict['hydrology_null_diffuse_till_water'] = ''

    # ocean
    ocean_params_dict = generate_ocean('null')

    # calving
    calving_params_dict = OrderedDict()
    #calving_params_dict = generate_calving('float_kill')

    exvars = params.read('exvars', 'str')
    exstep = params.read('exstep', 'float')
    tsstep = 'yearly'
    spatial_ts_dict = generate_spatial_ts(outfile, exvars, exstep, odir=os.path.join(outdir, tmp_dir), split=True)
    scalar_ts_dict = generate_scalar_ts(outfile, tsstep, odir=os.path.join(outdir, scalar_dir))

    all_params_dict = merge_dicts(general_params_dict,
                                  grid_params_dict,
                                  stress_balance_params_dict,
                                  climate_params_dict,
                                  ocean_params_dict,
                                  hydro_params_dict,
                                  calving_params_dict,
                                  spatial_ts_dict,
                                  scalar_ts_dict)
    all_params = ' '.join([' '.join(['-' + k, str(v)]) for k, v in all_params_dict.items()])

    cores = params.read('cores', 'int')
    mpido = 'mpirun -np {cores}'.format(cores=cores)
    pism_cmd = ' '.join([mpido, pism_prefix, all_params,
                         '> {outdir}/log_${outfile} 2>&1'.format(outdir=outdir, outfile=outfile), '\n'])

    extra_file = spatial_ts_dict['extra_file']
    myfiles = ' '.join(['{}_{:.3f}.nc'.format(extra_file, k) \
                        for k in np.arange(start_year + exstep, end_year, exstep)])
    myoutfile = extra_file + '.nc'
    myoutfile = os.path.join(outdir, spatial_dir, os.path.split(myoutfile)[-1])
    cmd_1 = ' '.join(['ncrcat -O -6 -h', myfiles, myoutfile])
    cmd_2 = ' '.join(['ncks -O -4', os.path.join(outdir, state_dir, outfile),
                      os.path.join(outdir, state_dir, outfile)])
    post_cmd = '\n'.join([cmd_1, cmd_2])
    post_cmd = post_cmd+'\n'

    return pism_cmd, post_cmd


def merge_dicts(*dict_args):
    '''
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    Returns: OrderedDict
    '''
    result = OrderedDict()
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def generate_grid_description(start_file, bootstrap=False):
    dem_data = Dataset(start_file, 'r')
    mx = len(dem_data.variables['x'][:])
    my = len(dem_data.variables['y'][:])
    dx = abs(dem_data.variables['x'][0]-dem_data.variables['x'][1])
    dem_data.close()

    horizontal_grid = OrderedDict()
    horizontal_grid['Mx'] = mx
    horizontal_grid['My'] = my

    if dx < 200:
        skip_max = 500
        mz = 101
        mzb = 21
    elif (dx >= 200) and (dx <= 500):
        skip_max = 250
        mz = 51
        mzb = 11
    else:
        skip_max = 100
        mz = 26
        mzb = 6

    vertical_grid = OrderedDict()
    vertical_grid['Lz'] = 2000
    vertical_grid['Lbz'] = 2000
    vertical_grid['z_spacing'] = 'equal'
    vertical_grid['Mz'] = mz
    vertical_grid['Mbz'] = mzb

    grid_options = OrderedDict()
    grid_options['skip'] = ''
    grid_options['skip_max'] = skip_max

    grid_dict = merge_dicts(horizontal_grid, vertical_grid, grid_options)

    if bootstrap:
        return grid_dict
    else:
        return grid_options

def generate_ocean(ocean, **kwargs):
    '''
    Generate ocean params
    Returns: OrderedDict
    '''

    params_dict = OrderedDict()
    if ocean in ['null']:
        pass
    elif ocean in ['const']:
        params_dict['ocean'] = 'constant'
    else:
        print('ocean {} not recognized, exiting'.format(ocean))
        import sys
        sys.exit(0)

    return merge_dicts(params_dict, kwargs)

def generate_calving(calving, **kwargs):
    '''
    Generate calving params
    Returns: OrderedDict
    '''

    params_dict = OrderedDict()
    if calving in ['ocean_kill']:
        params_dict['calving'] = calving
    elif calving in ['eigen_calving', 'vonmises_calving']:
        params_dict['calving'] = '{},thickness_calving'.format(calving)
    elif calving in ['hybrid_calving']:
        params_dict['calving'] = 'eigen_calving,vonmises_calving,thickness_calving'
    elif calving in ['float_kill']:
        params_dict['calving'] = calving
        params_dict['float_kill_margin_only'] = ''
    else:
        print('calving {} not recognized, exiting'.format(calving))
        import sys
        sys.exit(0)

    return merge_dicts(params_dict, kwargs)


def generate_spatial_ts(outfile, exvars, step, start=None, end=None, split=None, odir=None):
    '''
    Return dict to generate spatial time series
    Returns: OrderedDict
    '''

    # check if list or comma-separated string is given.
    try:
        exvars = ','.join(exvars)
    except:
        pass

    params_dict = OrderedDict()
    if split is True:
        outfile, ext = os.path.splitext(outfile)
        params_dict['extra_split'] = ''
    if odir is None:
        params_dict['extra_file'] = 'spatial_ts_' + outfile
    else:
        params_dict['extra_file'] = os.path.join(odir, 'spatial_ts_' + outfile)
    params_dict['extra_vars'] = exvars

    if step is None:
        step = 100

    if (start is not None and end is not None):
        times = '{start}:{step}:{end}'.format(start=start, step=step, end=end)
    else:
        times = step

    params_dict['extra_times'] = times


    return params_dict

def generate_scalar_ts(outfile, step, start=None, end=None, odir=None):
    '''
    Return dict to create scalar time series
    Returns: OrderedDict
    '''

    params_dict = OrderedDict()
    if odir is None:
        params_dict['ts_file'] = 'scalar_ts_' + outfile
    else:
        params_dict['ts_file'] = os.path.join(odir, 'scalar_ts_' + outfile)

    if step is None:
        step = 'yearly'

    if (start is not None and end is not None):
        times = '{start}:{step}:{end}'.format(start=start, step=step, end=end)
    else:
        times = step
    params_dict['ts_times'] = times

    return params_dict

if __name__ == '__main__':
    pism_cmd, post_cmd = generate_pism_cmd(params_file='test_params.txt')
    print pism_cmd
    print post_cmd