# IMPORTING NECESSARY LIBRARIES

import os
import numpy as np
import torch
import wget # pip3 install wget
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

def RunIter(height, width, focal_length, tform_cam2world,
             near_clip, far_clip, num_samples_per_ray,
             encoding_fn, batch_fn, model):
    
    # COMPUTE BUNDLE RAYS THROUGH ALL THE PIXELS 
    ray_origins, ray_directions = GetRayBundle(height, width, focal_length,
                                                 tform_cam2world)

    # GET THE SAMPLED QUERY POINTS AND DEPTH VALUES
    query_points, depth_values = QueryPointsFromRays(ray_origins, ray_directions,
                                                                near_clip, far_clip,
                                                                num_samples_per_ray)

    # FLATTEN 3D QUERY POINTS
    flattened_query_points = query_points.reshape((-1, 3))

    # ENCODE QUERY POINTS USING THE ENCODING FUNCTION
    encoded_points = encoding_fn(flattened_query_points)

    # SPLIT INTO MINIBATCHES AND RUN THE MODEL ON EACH MINIBATCH
    batches = batch_fn(encoded_points, chunksize=128)
    predictions = []
    for batch in batches:
        predictions.append(model(batch))
    radiance_field_flat = torch.cat(predictions, dim=0)

    # RESHAPE RADIANCE WITH AN ADDITIONAL DIMENSION OF SIZE 4 RGB AND DENSITY
    radiance_field = torch.reshape(radiance_field_flat, list(query_points.shape[:-1]) + [4])

    # SYNTHESIZE RGB IMAGE
    rgb_predicted, _, _ = RenderVolumeDensity(radiance_field, ray_origins, depth_values)

    return rgb_predicted