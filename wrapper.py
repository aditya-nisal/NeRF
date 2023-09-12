# IMPORTING NECESSARY LIBRARIES

import os
import numpy as np
import torch
import wget
import matplotlib.pyplot as plt
from Utils import *
from typing import Optional
from Network import *


# ALTERNATIVE IF THE DATASET IS NOT FOUND
if not os.path.exists('tiny_nerf_data.npz'):
    url = 'https://people.eecs.berkeley.edu/~bmild/nerf/tiny_nerf_data.npz'
    filename = wget.download(url)