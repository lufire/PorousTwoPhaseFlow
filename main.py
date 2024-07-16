# Main file to execute submodels. See imported script to be executed
import sys
import subprocess

# Specify dimensions ('1d', '2d')
dimensions = '1d'

if __name__ == '__main__':
    file_name = 'scripts/capillary_two_phase_flow_' + dimensions + '.py'
    subprocess.call([sys.executable, file_name])
