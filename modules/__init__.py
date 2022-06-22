import os, sys

modules_path = os.path.join(os.getcwd(), 'modules')
sys.path.append(modules_path)

from visualization import *
from clustering import *
from munging import *
from autoencoders import *