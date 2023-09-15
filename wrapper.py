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

def main():

    def RotX(angle): # CREATING ROTATION MATRIX ABOUT X
        angle = torch.tensor(angle)
        return torch.tensor([
            [1, 0, 0, 0],
            [0, torch.cos(angle), -torch.sin(angle), 0],
            [0, torch.sin(angle), torch.cos(angle), 0],
            [0, 0, 0, 1],
        ], dtype=torch.float32)

    def RotY(th): # CREATING ROTATION MATRIX ABOUT Y
        th = torch.tensor(th)
        return torch.tensor([
            [torch.cos(th), 0, -torch.sin(th), 0],
            [0, 1, 0, 0],
            [torch.sin(th), 0, torch.cos(th), 0],
            [0, 0, 0, 1],
        ], dtype=torch.float32)

    def PoseSpherical(th, angle, radius): # RETURNS CAMERA TO WORLD TRANSFORMATION MATRIX
        c2w = torch.Tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, radius],
            [0, 0, 0, 1],
        ])
        c2w = RotX(angle / 180 * np.pi) @ c2w
        c2w = RotY(th / 180 * np.pi) @ c2w
        c2w = torch.Tensor([
            [-1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ]) @ c2w
        return c2w

    names = [
        ['theta', [100, 0, 360]], # PARAMETERS FOR SPHERICAL POSE FUNCTION
        ['phi', [-30, -90, 0]],
        ['radius', [4, 3, 5]],
    ]
    import imageio
    f = 'video.mp4'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images, tform_cam2world, focal_length = LoadData()  # LOAD THE DATASET
    tform_cam2world = torch.from_numpy(tform_cam2world).to(device)  # CONVERT NUMPY ARRAY TO PYTORCH TENSOR AND MOVE THEM TO DEVICE
    focal_length = torch.from_numpy(focal_length).to(device)  # CONVERT NUMPY ARRAY TO PYTORCH TENSOR AND MOVE THEM TO DEVICE

    height, width = images.shape[1:3]  # EXTRACT IMAGE DIMENSIONS

    near_thresh = 2.  # NEAR CLIPPING PLANE
    far_thresh = 6.  # FAR CLIPPING PLANE

    testimg, testpose = images[101], tform_cam2world[101]  # SELECT TEST IMAGE AND CORRESPONDING POSE FOR EVALUATION
    testimg = torch.from_numpy(testimg).to(device)  # CONVERT NUMPY ARRAY TO PYTORCH TENSOR AND MOVE THEM TO DEVICE

    images = torch.from_numpy(images[:100, ..., :3]).to(device)  # CONVERT NUMPY ARRAY OF 1ST 100 ELEMENTS TO PYTORCH TENSOR AND MOVE THEM TO DEVICE

    num_encoding_functions = 6  # SET NUMBER OF ENCODING FUNCTIONS
    encode = lambda x: ComputePositionalEncoding(x,
                                                   num_encoding_functions=num_encoding_functions)  # ENCODING FUNCTION
    depth_samples_per_ray = 32  # DEFINE DEPTH SAMPLE PER RAY

    # OPTIMIZATION PARAMETERS
    lr = 5e-3  # LEARNING RATE
    num_iters = 1000  # NUM. ITERATIONS

    # MISCELLENEIOUS PARAMETERS
    display_every = 100  # AFTER HOW MANY ITERATIONS TRAINING STATS TO BE DISPLAYED

    model = Nerf(num_encoding_functions=num_encoding_functions)  # INITIALIZE THE MODEL
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # INITIALIZE ADAM OPTIMIZER WITH LEARNING RATE

    # STORE ERROR AND LOSS VALUES IN LIST
    list_of_error = []  
    iterations = []

    from tqdm import tqdm_notebook as tqdm

    angles = torch.linspace(0, 360, 300, dtype=torch.float32) # GENERATE 300 ANGLES BETWEEN 0-360

    for i in range(num_iters): # TRAINING LOOP

        target_img_idx = np.random.randint(images.shape[0])  # SELECT A RANDOM IMAGE INDEX
        target_img = images[target_img_idx].to(device)  # SELECT A RANDOM IMAGE
        target_tform_cam2world = tform_cam2world[target_img_idx].to(device)  # GET THE CORRESPONDING IMAGE POSE

        rgb_predicted = RunIter(height, width, focal_length,
                                 target_tform_cam2world, near_thresh,
                                 far_thresh, depth_samples_per_ray,
                                 encode, MiniBatches, model)  # FORWARD PASS OF THE MODEL YIELDING A RGB IMAGE

        loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)  # LOSS BETWEEN RGB VALUES AND ACTUAL IMAGE USING MSE
        loss.backward()  # BACKPROPOGATION
        optimizer.step()  # UPDATE MODEL PARAMS
        optimizer.zero_grad()  # RESET GRADIENTS

        if i % display_every == 0:
            rgb_predicted = RunIter(height, width, focal_length,
                                     testpose, near_thresh,
                                     far_thresh, depth_samples_per_ray,
                                     encode, MiniBatches, model)  # RUN THE MODEL
            loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)  # COMPUTE THE LOSS
            print("Loss Value:", loss.item())
            logloss = -10. * torch.log10(loss)

            list_of_error.append(logloss.item())
            iterations.append(i)

            plt.figure(figsize=(10, 4))
            plt.subplot(121)
            plt.imshow(rgb_predicted.detach().cpu().numpy())
            plt.title(f"{i}th iteration")
            plt.subplot(122)
            plt.plot(iterations, list_of_error)
            plt.title("Loss Plot")
            plt.show()

        print('Done!')

    # TEST THE MODEL
    images1 = []
    for th in angles:
        # COMPUTE CAMERA TO WORLD TRANSFORMATION MATRIX
        c2w = PoseSpherical(th, -30, 4).to(device)

        # RENDER THE SCENE USING TINY NERF
        rgb = RunIter(height, width, focal_length, c2w[:3, :4], 2, 6, depth_samples_per_ray, encode, MiniBatches,
                       model)

        # CONVER IMAGES TO NUMPY ARRAY 
        image = (255 * np.clip(rgb.clone().detach().cpu().numpy(), 0, 1)).astype(np.uint8)
        images1.append(image) # LIST OF RENDERING IMAGES USE TO CREATE A VIDEO
        plt.imshow(image)
        plt.show()

    # Write the list of images to a video file
    with imageio.get_writer(f, fps=27, quality=9) as writer:
        for image in images1:
            writer.append_data(image)


if __name__ == "__main__":
    main()