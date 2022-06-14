# Main file to execute submodels. See imported script to be executed
import sys
import subprocess

# Specify dimensions ('1d', '2d')
dimensions = '2d'

if __name__ == '__main__':
    file_name = 'src/capillary_two_phase_flow_' + dimensions + '.py'
    subprocess.call([sys.executable, file_name])
