import torch
import cv2

def CumulativeProdut(tensor):
    '''
    Takes: Tensor
    Returns: Exclusive cumulative produt of the inpu tensor
    '''
    cum_prod = torch.cumprod(tensor, dim = -1) #CUMPROD ALONG THE LAST DIMENSION

    rolled = torch.roll(cum_prod, 1, dims = -1) #SHIFTING THE CUMPROD TO ANOTHER AXIS

    rolled[..., 0] = 1 #FIRST ELEMENT IS 1

    exclusive_cum_prod = cum_prod / rolled

    return exclusive_cum_prod


def MiniBatches(inputs, chunksize = 1024*8):
    '''
    Takes: tensor to split, size of each tensor after splitting
    Returns: Minibatches after splitting
    '''
    return torch.split(inputs, chunksize) #SPLITTING THE TENSOR INTO BATCHES WITH EACH BATCH OF SIZE CHUNKSIZE


def NormalizeImage(image):
    return cv2.normalize(image,dst=None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)


def GetRayBundle(height, width, focal_length, T_cam2world):
    '''
    Takes: height of image, width of image, focal length of camera, camera to world transformation matrix
    Returns: Tuple containing the ray origins and directions
    '''

    #COMPUTING MESHGRID OF THE PIXELS
    w = torch.arange(width, dtype=torch.float32, device = T_cam2world.device)
    h = torch.arange(height, dtype=torch.float32, device = T_cam2world.device)
    w, h = torch.meshgrid(w, h)
    ww, hh = w.transpose(-1, -2), h.transpose(-1, -2) #INTERCHANGING THE LAST AND THE SECOND TO LAST DIMENSIONS. THIS IS DONE FOR A BETTER REPRESENTATION OF A 3D POINT

    #COMPUTING PIXEL DIRECTIONS. THIS IS DONE TO SET THE CENTER PIXEL TO (0, 0) AND OTHERS SYMMETRIC ESPECT TO IT 
    pixel_directions = torch.stack(
        [
            (ww - width * .5) / focal_length,
            -(hh - height * .5) / focal_length,
            -torch.ones_like(ww),
        ],
        dim=-1,
    )

    #PIXEL DIRECTINS TO WORLD DIRECTIONS
    world_directions = torch.sum(pixel_directions[..., None, :] * T_cam2world[:3, :3], dim = -1)
    ray_origins = T_cam2world[:3, -1].expand(world_directions.shape)
    return ray_origins, world_directions


def QueryPointsFromRays(ray_origins, ray_directions, near_plane, far_plane, num_samples, randomize=True):
    """
    Takes:
    Ray_origins: Origin of each ray in the "bundle" as returned by the `GetRayBundle()` method.
    Ray_directions: Direction of each ray in the "bundle" as returned by the `GetRayBundle()` method.
    Near_plane: Near plane for the depth values.
    Far_plane: Far plane for the depth values.
    Num_samples: Number of depth samples along each ray.
    Randomize: Whether to randomize the depth samples.
    Returns: 
    Tuple containing the query points and depth values.
    """
    device = ray_origins.device # GET DEVICE OF RAY ORIGIN
    batch_size = ray_origins.shape[0] # GET BATCH SIZE
    num_rays = ray_origins.shape[1] # GET NO. OF RAYS

    # GENERATE LIST OF DEPTH VALUE FOR EACH RAY AND LOOP IT
    depth_values_list = []
    for _ in range(batch_size):
        for _ in range(num_rays):
            depth_values = torch.linspace(near_plane, far_plane, num_samples, device=device) # CREATE LIST OF DEPTH VALUES
            if randomize: 
                noise = torch.rand((num_samples,), device=device) # GENERATING A LIST OF RANDOMNESS
                depth_values += noise * (far_plane - near_plane) / num_samples # ADDING RANDOMNESS TO DEPTH VALUE
            depth_values_list.append(depth_values) 

    # STACK DEPTH_VALUES AS TENSOR AND RESHAPE TO *BAATCH_SIZE, NUM_RAYS, NUM_SAMPLES)
    depth_values = torch.stack(depth_values_list, dim=0).reshape(batch_size, num_rays, num_samples)
    
    # BASICALLY TRANSLATION + ROTATION INTO DIRECTION. GETS EVERY POINT THROUGH THE NUM_SAMPLES WE PASSED
    query_points = ray_origins[..., None, :] + ray_directions[..., None, :] * depth_values[..., :, None]

    return query_points, depth_values # GET QUERY POINTS AND THE DEPTH VALUES


def RenderVolumeDensity(radiance_field, ray_origins, depth_values):
    """
    Takes:
    Radiance field of the volume.
    Origin of each ray in the "bundle" as returned by the `GetRayBundle()` method.
    Depth values along each ray as returned by the `QueryPointsFromRays()` method.
    Returns:
    Tuple containing the density map, depth map, and accumulated density map.
    """
    attenuation = torch.nn.functional.relu(radiance_field[..., 3]) # GET THE ATTENUATION VALUES
    color = torch.sigmoid(radiance_field[..., :3]) # GET THE COLOUR VALUES
    max_depth = torch.tensor([1e10], dtype=ray_origins.dtype, device=ray_origins.device) # GET MAXIMUM DEPTH VALUES
    ray_lengths = torch.cat((depth_values[..., 1:] - depth_values[..., :-1], max_depth.expand(depth_values[..., :1].shape)), dim=-1) # COMPUTE LENGTH OF EACH RAY SEGMENT
    ray_alphas = 1. - torch.exp(-attenuation * ray_lengths) # COMPUTE ALPHA VALUES FOR EACH RAY SEGMENT
    ray_weights = ray_alphas * CumulativeProdut(1. - ray_alphas + 1e-10) # COMPUTE WEIGHTS FOR EACH RAY SEGMENT
    color_map = (ray_weights[..., None] * color).sum(dim=-2) # GET TEH COLOUR MAP
    depth_map = (ray_weights * depth_values).sum(dim=-1) # GET THE DEPTH MAP
    weight_sum = ray_weights.sum(-1) # COMPUTE ACCUMULATED DENSITY MAP
    return color_map, depth_map, weight_sum # RETURN COLOUR MAP, DEPTH MAP AND ACCUMULATED DENSITY MAP


def ComputePositionalEncoding(input_tensor, num_encoding_functions=6, include_input=True, use_log_sampling=True):
    """
    Takes:
    Input tensor.
    Number of encoding functions to use.
    Whether to include the input tensor in the encoding.
    Whether to use logarithmic sampling of the encoding functions.
    Returns: 
    Positional encoding tensor.
    """

    encoding = [input_tensor] if include_input else [] # INITIALIZE THE ENCODING LIST

    device = input_tensor.device
    
    # COMPUTE THE FREQUENCY BAND
    if use_log_sampling:      # FREQUENCIES ARE GENERATED IN LOGARITHMIC SCALE (FREQ^2)
        freq_bands = 2.0 ** torch.linspace( 
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=input_tensor.dtype,
            device=device,
        )
    else:                     # FREQUENCIES ARE LINEARLY SPACED BETWEEN GIVEN RANGE
        freq_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=input_tensor.dtype,
            device=device,
        )

    for freq in freq_bands:
        encoding.append(torch.sin(input_tensor * freq))
        encoding.append(torch.cos(input_tensor * freq))

    return torch.cat(encoding, dim=-1) # RETURN THE CONCATENATED INDIVIDUAL ENCODED TENSORS