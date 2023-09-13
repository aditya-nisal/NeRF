# IMPORTING NECESSARY LIBRARIES

import os
import numpy as np
import torch
import wget
import matplotlib.pyplot as plt
from utils import *
from typing import Optional
from network import *


# ALTERNATIVE IF THE DATASET IS NOT FOUND
if not os.path.exists('tiny_nerf_data.npz'):
    url = 'https://people.eecs.berkeley.edu/~bmild/nerf/tiny_nerf_data.npz'
    filename = wget.download(url)

# FUNCTIONTO LOAD THE DATASET FOR TINY NERF

def LoadData():
    '''
    Takes: -
    Returns:
        images, poses and focal length
    '''
    data = np.load('tiny_nerf_data.npz')
    images = data['images'].astype(np.float32)
    poses = data['poses'].astype(np.float32)
    focal = np.array(data["focal"])

    return images, poses, focal