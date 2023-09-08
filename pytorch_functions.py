import torch
import math

x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]).view(4, 2)
print("torch.tensor: {}".format(x)) #CREATES TENSOR

print("torch.split: {}".format(torch.split(x, 2))) #SPLITS TENSOR INTO NMBR OG GROUPS

print("torch.cumprod: {}".format(torch.cumprod(x, dim = -1))) #PRODUCT OF THE ELEMENTS OF TENSOR ALONG A GIVEN AXIS

print("torch.roll: {}".format(torch.roll(x, 3, dims = -1))) #ROLLS THE COLUMNS OR ROWS AS PER THE AXIS

print(x.shape)
print("torch.cat: {}".format(torch.cat((torch.ones((x.shape[0], x.shape[1])), x), dim=-1))) #CONCATENETES THE TENSOR ALONG THE SPECIFIED AXIS

print("torch.arange: {}".format(torch.arange(0, 10, 1, dtype = torch.float32))) #SIMILAR TO np.arange

A = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
B = torch.tensor([1, 2, 3, 4, 5, 6])
print("torch.meshgrid: {}".format(torch.meshgrid(A, B))) #CREATES A MATRIX OF SIZE LEN(A), LEN(B)

C = torch.tensor([7, 8, 9, 10, 11, 12])
print("torch.stack: {}".format(torch.stack((B, C), dim = 0))) #SIMILAR TO CONCATENATION

print("torch.linspace: {}".format(torch.linspace(1, 10, steps = 10))) #SIMILAR TO np.linspace

print("torch.rand: {}".format(torch.rand(2, 3, 2))) #CREATES A RANDOM ARRAY OR LIST OF GIVEN SHAPE

print("torch.sigmoid: {}".format(torch.sigmoid(torch.rand(3, 3, 3)))) #WORKS AS A SIGMOID (SQUEEZE THE OUTPUT BETWEEN 0 AND 1)

print("torch.exp: {}".format(torch.exp(torch.tensor([0, math.log(2.)])))) #exp OF WHATEVER TENSOR IT PROVIDED

angle_degrees = torch.tensor([0, 30, 45, 60, 90], dtype = torch.float32)
angle_radians = torch.deg2rad(angle_degrees) #CONVERTING EDGREES TENSOR TO RADIANS
print("torch.sin: {}".format(torch.sin(angle_radians))) #SIN OF A RADIAN ANGLE
print("torch.cos: {}".format(torch.cos(angle_radians))) #COS OF A RADIAN ANGLE
