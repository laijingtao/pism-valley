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

class PISMConfig(object):
    def __init__(self, params_file=None):
        self.params_file = params_file
        self.params = ParamParser(params_file)
        self.header = self.make_batch_header()
        self.pism_cmd, self.post_cmd = self.generate_pism_cmd()
        self.ela_cmd = self.generate_ela_cmd()


    def generate_pism_cmd(self):
        try:
            pism_exec = os.path.join(os.environ['PISM_PREFIX'], 'pismr')
        except:
            pism_exec = 'pismr'

        system = self.params.read('system')
        cores = self.params.read('cores', 'int')
        walltime = self.params.read('walltime', 'str')
        if system in ['keeling']:
            pism_data_dir = os.environ['PISM_DATA_DIR']
        else:
            pism_data_dir = './'

        outdir = self.params.read('outdir')
        outdir = os.path.join(pism_data_dir, outdir)
        self.outdir = outdir
        state_dir = 'state'
        scalar_dir = 'scalar'
        spatial_dir = 'spatial'
        erosion_dir = 'erosion'
        postproc_dir = 'postproc'
        initial_dir = 'initial'
        tmp_dir = 'tmp'
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        for subdir in [scalar_dir, spatial_dir, state_dir, erosion_dir, postproc_dir, initial_dir, tmp_dir]:
            if not os.path.isdir(os.path.join(outdir, subdir)):
                os.mkdir(os.path.join(outdir, subdir))

        general_params_dict = OrderedDict()
        start_file = self.params.read('start_file', 'str')
        general_params_dict['i'] = start_file
        bootstrap = self.params.read('bootstrap', 'bool')
        if bootstrap:
            general_params_dict['bootstrap'] = ''
        start_year = self.params.read('start_year', 'float')
        end_year = self.params.read('end_year', 'float')
        outfile = self.generate_outfile_name()
        outfile_base = os.path.splitext(outfile)[0]
        general_params_dict['ys'] = start_year
        general_params_dict['ye'] = end_year
        general_params_dict['o'] = os.path.join(outdir, state_dir, outfile)
        general_params_dict['o_format'] = 'netcdf4_parallel'
        general_params_dict['o_size'] = 'small'
        general_params_dict['options_left'] = ''

        grid_params_dict = self.generate_grid_description(start_file, bootstrap)

        # stress balance
        stress_balance_params_dict = OrderedDict()
        stress_balance = self.params.read('stress_balance', 'str')
        stress_balance_params_dict['stress_balance'] = stress_balance
        # sia
        #stress_balance_params_dict['sia_flow_law'] = 'gpbld3' # Note: gpbld3 will trigger KSP solver error
        stress_balance_params_dict['sia_flow_law'] = 'gpbld'
        stress_balance_params_dict['sia_n'] = 3.0
        stress_balance_params_dict['sia_e'] = self.params.read('sia_e', 'float')
        stress_balance_params_dict['bed_smoother_range'] = 50
        # sia_sliding
        if stress_balance == 'weertman_sliding+sia':
            stress_balance_params_dict['stress_balance.weertman_sliding.f'] = self.params.read('f_sliding', 'float')
        # ssa
        if stress_balance == 'ssa+sia':
            stress_balance_params_dict['ssa_e'] = self.params.read('ssa_e', 'float')
            stress_balance_params_dict['ssa_n'] = 3.0
            stress_balance_params_dict['pseudo_plastic'] = ''
            stress_balance_params_dict['pseudo_plastic_q'] = self.params.read('pseudo_plastic_q', 'float')
            stress_balance_params_dict['till_effective_fraction_overburden'] = self.params.read('till_effective_fraction_overburden', 'float')
            stress_balance_params_dict['plastic_phi'] = self.params.read('plastic_phi', 'float')
            stress_balance_params_dict['ssa_method'] = 'fd'
            #stress_balance_params_dict['ssafd_ksp_divtol'] = 1e300
            #stress_balance_params_dict['cfbc'] = ''

        # climate
        climate_params_dict = OrderedDict()
        ice_density = 910.
        climate_params_dict['surface'] = 'pdd'
        climate_params_dict['surface.pdd.factor_ice'] = 4.59 / ice_density  # Shea et al (2009)
        climate_params_dict['surface.pdd.factor_snow'] = 3.04 / ice_density  # Shea et al (2009)
        climate_params_dict['surface.pdd.refreeze'] = 0

        air_temp_mean_annual = self.params.read('air_temp_mean_annual', 'float')
        air_temp_mean_july = air_temp_mean_annual+5
        precipitation = self.params.read('precipitation', 'float')
        climate_file = os.path.join(outdir, initial_dir, 'climate_{}.nc'.format(outfile_base))
        generate_constant_climate_file(infile=start_file,
                                       outfile=climate_file,
                                       air_temp_mean_annual=air_temp_mean_annual,
                                       air_temp_mean_july=air_temp_mean_july,
                                       precipitation=precipitation)
        temp_lapse_rate = self.params.read('temp_lapse_rate', 'float')
        climate_params_dict['atmosphere'] = 'yearly_cycle,lapse_rate'
        climate_params_dict['atmosphere_yearly_cycle_file'] = climate_file
        climate_params_dict['atmosphere_lapse_rate_file'] = climate_file
        climate_params_dict['temp_lapse_rate'] = temp_lapse_rate

        do_cycle = self.params.read('cycle', 'bool')
        if do_cycle:
            atmosphere_paleo_file = self.params.read('atmosphere_paleo_file', 'str')
            climate_params_dict['atmosphere'] += ',delta_T,paleo_precip'
            climate_params_dict['atmosphere_delta_T_file'] = atmosphere_paleo_file
            #climate_params_dict['atmosphere_delta_T_period'] = 100000
            #climate_params_dict['atmosphere_delta_T_reference_year'] = 0
            climate_params_dict['atmosphere_paleo_precip_file'] = atmosphere_paleo_file
            #climate_params_dict['atmosphere_paleo_precip_period'] = 100000
            #climate_params_dict['atmosphere_paleo_precip_reference_year'] = 0

        # bed deformation
        beddef_params_dict = OrderedDict()
        do_erosion = self.params.read('erosion', 'bool')
        if do_erosion:
            beddef_params_dict['erosion'] = ''
            beddef_params_dict['erosion_k'] = self.params.read('erosion_k', 'float')

        # hydrology diffuse
        hydro_params_dict = OrderedDict()
        hydro_params_dict['hydrology'] = 'null'
        hydro_params_dict['hydrology_null_diffuse_till_water'] = ''

        # ocean
        ocean_params_dict = self.generate_ocean('null')

        # calving
        calving_params_dict = OrderedDict()
        #ocean_kill_file = os.path.join(outdir, initial_dir, 'calving_ocean_kill_{}.nc'.format(outfile_base))
        #generate_ocean_kill_file(infile=start_file, outfile=ocean_kill_file)
        #calving_params_dict = self.generate_calving('ocean_kill', ocean_kill_file=ocean_kill_file)
        #calving_params_dict = self.generate_calving('float_kill')

        exvars = self.params.read('exvars', 'str')
        if stress_balance == 'weertman_sliding+sia':
            exvars.remove('tauc')
        exstep = self.params.read('exstep', 'float')
        tsstep = 'yearly'
        spatial_ts_dict = self.generate_spatial_ts(outfile, exvars, exstep, odir=os.path.join(outdir, tmp_dir), split=True)
        scalar_ts_dict = self.generate_scalar_ts(outfile, tsstep, odir=os.path.join(outdir, scalar_dir))

        all_params_dict = self.merge_dicts(general_params_dict,
                                           grid_params_dict,
                                           stress_balance_params_dict,
                                           climate_params_dict,
                                           beddef_params_dict,
                                           hydro_params_dict,
                                           ocean_params_dict,
                                           calving_params_dict,
                                           spatial_ts_dict,
                                           scalar_ts_dict)
        all_params = ' '.join([' '.join(['-' + k, str(v)]) for k, v in all_params_dict.items()])

        mpido = 'mpirun -np {cores}'.format(cores=cores)
        pism_cmd = ' '.join([mpido, pism_exec, all_params, '>',
            os.path.join(outdir, 'log_{}_id_${}.log 2>&1'.format(outfile_base, 'SLURM_JOB_ID')), '\n'])

        self.general_params_dict = general_params_dict
        self.grid_params_dict = grid_params_dict
        self.stress_balance_params_dict = stress_balance_params_dict
        self.climate_params_dict = climate_params_dict
        self.beddef_params_dict = beddef_params_dict
        self.hydro_params_dict = hydro_params_dict
        self.ocean_params_dict = ocean_params_dict
        self.calving_params_dict = calving_params_dict
        self.spatial_ts_dict = spatial_ts_dict
        self.scalar_ts_dict = scalar_ts_dict

        extra_file = spatial_ts_dict['extra_file']
        tmp_files = ' '.join(['{}_{:.3f}.nc'.format(extra_file, k) \
                            for k in np.arange(start_year+exstep, end_year+exstep, exstep)])
        spatial_outfile = extra_file + '.nc'
        spatial_outfile = os.path.join(outdir, spatial_dir, os.path.split(spatial_outfile)[-1])
        cmd_1 = ' '.join(['ncrcat -O -6 -h', tmp_files, spatial_outfile])
        cmd_2 = ' '.join(['ncks -O -4', os.path.join(outdir, state_dir, outfile),
                          os.path.join(outdir, state_dir, outfile)])
        #post_cmd = '\n'.join([cmd_1, cmd_2])
        post_cmd = cmd_1
        post_cmd = post_cmd+'\n'
        

        cmd = ['cp', self.params_file, os.path.join(outdir, initial_dir, '{}_params.txt'.format(outfile_base))]
        sub.call(cmd)
        cmd = ['cp', start_file, os.path.join(outdir, initial_dir, 'initial_{}.nc'.format(outfile_base))]
        sub.call(cmd)

        return pism_cmd, post_cmd


    def merge_dicts(self, *dict_args):
        '''
        Given any number of dicts, shallow copy and merge into a new dict,
        precedence goes to key value pairs in latter dicts.
        Returns: OrderedDict
        '''
        result = OrderedDict()
        for dictionary in dict_args:
            result.update(dictionary)
        return result

    def generate_outfile_name(self):
        abbr = {'stress_balance': 'sb',
                'air_temp_mean_annual': 'T',
                'precipitation': 'P',
                'start_year': 's',
                'end_year': 'e'}

        outfile = self.params.read('outfile', 'str')
        outfile = os.path.splitext(outfile)[0]
        name_options = self.params.read('name_options', 'str')
        name_dict = OrderedDict()
        for key in name_options:
            name_dict[abbr[key]] = self.params.read(key, 'str')
        postfix = '_'.join([k+'_'+v for k, v in name_dict.items()])
        outfile = outfile + '_' + postfix + '.nc'

        return outfile

    def generate_grid_description(self, start_file, bootstrap=False):
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
        vertical_grid['Lz'] = 5000
        vertical_grid['Lbz'] = 2000
        vertical_grid['z_spacing'] = 'equal'
        vertical_grid['Mz'] = mz
        vertical_grid['Mbz'] = mzb

        grid_options = OrderedDict()
        grid_options['skip'] = ''
        grid_options['skip_max'] = skip_max

        grid_dict = self.merge_dicts(horizontal_grid, vertical_grid, grid_options)

        if bootstrap:
            return grid_dict
        else:
            return grid_options

    def generate_ocean(self, ocean, **kwargs):
        '''
        Generate ocean params
        Returns: OrderedDict
        '''

        params_dict = OrderedDict()
        if ocean in ['null']:
            pass
        elif ocean in ['dry']:
            params_dict['dry'] = ''
        elif ocean in ['const']:
            params_dict['ocean'] = 'constant'
        else:
            print('ocean {} not recognized, exiting'.format(ocean))
            import sys
            sys.exit(0)

        return self.merge_dicts(params_dict, kwargs)

    def generate_calving(self, calving, **kwargs):
        '''
        Generate calving params
        Returns: OrderedDict
        '''

        params_dict = OrderedDict()
        if calving in ['ocean_kill']:
            params_dict['calving'] = calving
            params_dict['ocean_kill_file'] = kwargs['ocean_kill_file']
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

        return self.merge_dicts(params_dict, kwargs)


    def generate_spatial_ts(self, outfile, exvars, step, start=None, end=None, split=None, odir=None):
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

    def generate_scalar_ts(self, outfile, step, start=None, end=None, odir=None):
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

    def generate_ela_cmd(self):
        start_year = self.params.read('start_year', 'float')
        end_year = self.params.read('end_year', 'float')
        ela_step = self.params.read('ela_step', 'float')

        ela_cmd = '\n'
        for k in np.arange(start_year+ela_step, end_year+ela_step, ela_step):
            tmp_ela_cmd = self.generate_ela_cmd_at_time(time=k)
            ela_cmd += tmp_ela_cmd+'\n'

        return ela_cmd
    

    def generate_ela_cmd_at_time(self, time=None):
        if time is None:
            sys.exit("Missing time")
        cores = self.params.read('cores', 'int')
        mpido = 'mpirun -np {cores}'.format(cores=cores)
        try:
            pism_exec = os.path.join(os.environ['PISM_PREFIX'], 'pismr')
        except:
            pism_exec = 'pismr'
        pism_prefix = ' '.join([mpido, pism_exec])

        outdir = self.outdir
        outfile = self.generate_outfile_name()
        file_name_base = os.path.splitext(outfile)[0]
        start_file = 'spatial_ts_{}_{:.3f}.nc'.format(file_name_base, time)
        start_file = os.path.join(outdir, 'tmp', start_file)

        other_dict = self.merge_dicts(self.stress_balance_params_dict,
                                      self.climate_params_dict,
                                      self.ocean_params_dict,
                                      self.hydro_params_dict,
                                      self.calving_params_dict)

        ela_params_dict = OrderedDict()
        ela_params_dict['i'] = start_file
        ela_params_dict['bootstrap'] = ''
        ela_params_dict['ys'] = time
        ela_params_dict['ye'] = time+1.0
        ela_params_dict['o'] = os.path.join(outdir, 'tmp', 'mb_{}_at_{}.nc'.format(file_name_base, time))
        ela_params_dict['extra_file'] = os.path.join(outdir, 'tmp', 'mb_ts_{}_at_{}'.format(file_name_base, time))
        ela_params_dict['extra_split'] = ''
        ela_params_dict['extra_times'] = 0.1
        ela_params_dict['extra_vars'] = 'climatic_mass_balance,thk,usurf'

        if other_dict is not None:
            ela_params_dict = self.merge_dicts(ela_params_dict, other_dict)

        ela_params = ' '.join([' '.join(['-' + k, str(v)]) for k, v in ela_params_dict.items()])

        pism_cmd = ' '.join([pism_prefix, ela_params, '>',
                             os.path.join(outdir, 'tmp', 'log_mb_{}_at_{}.log 2>&1'.format(file_name_base, time)), '\n'])

        extra_file = ela_params_dict['extra_file']
        exstep = ela_params_dict['extra_times']
        tmp_files = ' '.join(['{}_{:.3f}.nc'.format(extra_file, k) \
                            for k in np.arange(time+exstep, time+1.0, exstep)])
        spatial_outfile = extra_file + '.nc'
        spatial_outfile = os.path.join(outdir, 'postproc', 'mb_ts_{}_at_{}.nc'.format(file_name_base, time))
        cmd = ' '.join(['ncrcat -O -6 -h', tmp_files, spatial_outfile])
        cmd = cmd + '\n'

        ela_cmd = '\n'.join([pism_cmd, cmd])

        return ela_cmd

    def make_batch_header(self):
        system = self.params.read('system', 'str')
        cores = self.params.read('cores', 'int')
        walltime = self.params.read('walltime', 'str')
        if cores > 12:
            partition = 'sesempi'
        else:
            partition = 'node'

        batch_system_list = {}

        batch_system_list['debug'] = {'job_id': 'test'}
        batch_system_list['debug']['header'] = '#!/bin/bash\n'

        batch_system_list['keeling'] = {'job_id': 'SLURM_JOBID'}
        header = """#!/bin/bash
#SBATCH -n {cores}
#SBATCH --mem-per-cpu=4096
#SBATCH -p {partition}
#SBATCH --time={walltime}
#SBATCH --mail-user=jlai11@illinois.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --export=PATH,LD_LIBRARY_PATH

module list
cd $SLURM_SUBMIT_DIR

""".format(cores=cores, partition=partition, walltime=walltime)
        batch_system_list['keeling']['header'] = header

        return batch_system_list[system]['header']


def generate_constant_climate_file(infile=None, outfile=None, *args, **kwargs):
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

    climate_write_data(infile, outfile, air_temp_mean_annual=air_temp_mean_annual,
               air_temp_mean_july=air_temp_mean_july, precipitation=precipitation,
               usurf=usurf)

def climate_write_data(infile=None, outfile=None, *args, **kwargs):
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
    

def generate_ocean_kill_file(infile=None, outfile=None, *args, **kwargs):
    try:
        thk = kwargs['thk']
    except:
        thk = 0

    if not(infile is None):
        indata = Dataset(infile, 'r')
        x_dim, = indata.variables['x'].shape
        y_dim, = indata.variables['y'].shape
        indata.close()

    topg = np.zeros((y_dim, x_dim))
    topg[:] = 1
    topg[:, 0][:] = -1
    topg[:, x_dim-1][:] = -1
    
    calving_write_data(infile, outfile, thk=thk, topg=topg)


def calving_write_data(infile=None, outfile=None, *args, **kwargs):
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

    topg = kwargs['topg']
    thk = kwargs['thk']

    dim_len = {'x': x_dim, 'y': y_dim, 'time': None}
    for dim_name in ['x', 'y', 'time']:
        try:
            outdata.createDimension(dim_name, dim_len[dim_name])
            outdata.createVariable(dim_name, np.float64, (dim_name,))
            if not(dim_len[dim_name] is None):
                outdata.variables[dim_name][:] = dim_value[dim_name]
        except:
            continue

    var_name_list = ['topg', 'thk']
    var_value = {'topg': topg,
                 'thk': thk}
    var_unit = {'topg': 'm',
                'thk': 'm'}
    var_standard_name = {'topg': 'bedrock_altitude',
                         'thk': 'land_ice_thickness'}
    for var_name in var_name_list:
        try:
            var = outdata.variables[var_name]
        except:
            var = outdata.createVariable(var_name, np.float64, ('y', 'x',))
        var[:] = var_value[var_name]
        var.units = var_unit[var_name]
        var.standard_name = var_standard_name[var_name]

    outdata.close()
    
def generate_uplift_file(infile=None, outfile=None, *args, **kwargs):
    try:
        uplift_rate = kwargs['uplift_rate']
    except:
        sys.exit('Miss uplift rate')

    dim_value = {}
    if not(infile is None):
        indata = Dataset(infile, 'r')
        x_dim, = indata.variables['x'].shape
        y_dim, = indata.variables['y'].shape
        dim_value['x'] = indata.variables['x'][:]
        dim_value['y'] = indata.variables['y'][:]
        indata.close()

    uplift = np.zeros((y_dim, x_dim))
    uplift[:] = uplift_rate

    outdata = Dataset(outfile, 'w')

    dim_len = {'x': x_dim, 'y': y_dim, 'time': None}
    for dim_name in ['x', 'y', 'time']:
        try:
            outdata.createDimension(dim_name, dim_len[dim_name])
            outdata.createVariable(dim_name, np.float64, (dim_name,))
            if not(dim_len[dim_name] is None):
                outdata.variables[dim_name][:] = dim_value[dim_name]
        except:
            continue

    var_name_list = ['fixed_uplift']
    var_value = {'fixed_uplift': uplift}
    var_unit = {'fixed_uplift': 'm year-1'}
    var_standard_name = {'fixed_uplift': 'fixed_uplift_rate'}
    for var_name in var_name_list:
        try:
            var = outdata.variables[var_name]
        except:
            var = outdata.createVariable(var_name, np.float64, ('y', 'x',))
        var[:] = var_value[var_name]
        var.units = var_unit[var_name]
        var.standard_name = var_standard_name[var_name]

    outdata.close()

if __name__ == '__main__':
    pism_cmd, post_cmd = generate_pism_cmd(params_file='test_params.txt')
    print pism_cmd
    print post_cmd
