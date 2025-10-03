import subprocess
import sys

subprocess.Popen(['start', 'cmd', '/K', 'python', 'model_training.py'], shell=True)