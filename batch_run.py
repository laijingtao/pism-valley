import os
import sys

sys.path.append(os.environ['PARAMPARSER_PATH'])
from paramparser import ParamParser
from generate_pism_cmd import generate_pism_cmd

params_file = 'test_params.txt'
params = ParamParser(params_file)
system = params.read('system', 'str')

pism_cmd, post_cmd = generate_pism_cmd(params_file=params_file)

with open('test_run.sh', 'w') as f:
    if system == 'keeling':
        header = """#!/bin/bash
#SBATCH -n {cores}
#SBATCH --mem-per-cpu=4096
#SBATCH --time={walltime}
#SBATCH --mail-user=jlai11@illinois.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --export=PATH,LD_LIBRARY_PATH

module list
cd $SLURM_SUBMIT_DIR
""".format(cores=params.read('cores', 'int'),
           walltime=params.read('walltime', 'str'))
        f.write(header+'\n')
    f.write(pism_cmd)
    f.write(post_cmd)
