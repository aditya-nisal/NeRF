import torch
x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]).view(4, 2)
print("x: {}".format(x))
print("get_minibatches: {}".format(torch.split(x, 2)))
print("cum_prod: {}".format(torch.cumprod(x, dim = 1)))
print("roll: {}".format(torch.roll(x, 1, dims = -1)))