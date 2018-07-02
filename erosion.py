import sys
import os
sys.path.append(os.environ['PARAMPARSER_PATH'])
from paramparser import ParamParser
from generate_pism_cmd import generate_pism_cmd
from generate_pism_cmd import generate_outfile_name
try:
    import subprocess32 as sub
except:
    import subprocess as sub

domain = 'valley'

param_file = '{}_erosion_params.txt'.format(domain)
params = ParamParser(param_file)

start_year = 0
end_year = 50000
erosion_dt = 100
nt = int((end_year-start_year)/100)

sb_methods = ['ssa+sia']
T_list = ['2.0']
P_list = ['1000.0']

script_list = []
if not os.path.isdir('./run'):
    os.mkdir('./run')

for T in T_list:
    for P in P_list:
        for sb in sb_methods:
            update_topg_path = os.path.abspath('./update_topg.py')
            erosion_file = None
            for i in range(nt):
                t = (i+1)*erosion_dt+start_year
                
                outdir = params.read('outdir', 'str')
                pism_data_dir = os.environ['PISM_DATA_DIR']
                outdir = os.path.join(pism_data_dir, outdir)
                if not os.path.isdir(outdir):
                    os.mkdir(outdir)
                for subdir in ['tmp']:
                    if not os.path.isdir(os.path.join(outdir, subdir)):
                        os.mkdir(os.path.join(outdir, subdir))

                script_name = os.path.join(outdir, 'tmp', 
                    'run_erosion_{}_{}_T_{}_P_{}_s_{}_e_{}.sh'.format(domain, sb, T, P, t-erosion_dt, t))
                tmp_params_file = os.path.join(outdir, 'tmp',
                    '{}_params_{}_T_{}_P_{}_s_{}_e_{}.txt'.format(domain, sb, T, P, t-erosion_dt, t))
                params.change_value('exstep', '10', tmp_params_file)
                params.change_value('start_year', t-erosion_dt, tmp_params_file)
                params.change_value('end_year', t, tmp_params_file)
                if i > 0:
                    params.change_value('start_file', erosion_file, tmp_params_file)
                    params.change_value('bootstrap', 'False', tmp_params_file)

                params.change_value('stress_balance', sb, tmp_params_file)
                params.change_value('air_temp_mean_annual', T, tmp_params_file)
                params.change_value('precipitation', P, tmp_params_file)

                header, pism_cmd, post_cmd, ela_cmd = generate_pism_cmd(params_file=tmp_params_file)
                with open(script_name, 'w') as f:
                    f.write(pism_cmd)
                    f.write(post_cmd)
                    f.write(ela_cmd)
                sub.call(['bash', script_name])

                outfile = generate_outfile_name(params)
                state_file = os.path.join(outdir, 'state', outfile)
                spatial_file = os.path.join(outdir, 'spatial', 'spatial_ts_'+outfile)
                erosion_file = os.path.join(outdir, 'erosion', 'erosion_'+outfile)
                cmd = ['python', update_topg_path, '--state_file', state_file,
                       '--spatial_file', spatial_file, '--outfile', erosion_file,
                       '--dt', str(erosion_dt)]
                sub.call(cmd)

                tmp_dir = os.path.join(outdir, 'tmp')
                for tmp_file in os.listdir(tmp_dir):
                    try:
                        os.remove(os.path.join(tmp_dir, tmp_file))
                    except OSError:
                        pass

print 'Done!!!'
