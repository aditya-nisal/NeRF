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
