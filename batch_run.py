import os
import sys

sys.path.append(os.environ['PARAMPARSER_PATH'])
from paramparser import ParamParser
from generate_pism_cmd import generate_pism_cmd

params_file = 'test_params.txt'

pism_cmd, post_cmd = generate_pism_cmd(params_file=params_file)

with open('test_run.sh', 'w') as f:
    f.write(pism_cmd)
    f.write(post_cmd)