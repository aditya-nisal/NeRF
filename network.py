import torch.nn as nn
import torch.nn.functional as F
import torch

########################################## ACTUAL NERF ##########################################

class Actual_NeRF(nn.Module):
    def __init__(self, pos_embd_size, W, dir_embd_size): # pos_embd_size, W, AND dir_embd_size ARE THE INPUT PARAMETER 
        super(Actual_NeRF, self).__init__() # CALLS THE INITIALIZATION METHOD OF THE nn.Module

# STACK VARIOUS NERUAL NETWORK LAYERS USING nn.Sequential

        self.Layers1 = nn.Sequential(
            nn.Linear(pos_embd_size * 6 + 3, W),
            nn.ReLU(),
            nn.Linear(W, W),
            nn.ReLU(),
            nn.Linear(W, W),
            nn.ReLU(),
            nn.Linear(W, W),
            nn.ReLU(),
        )

        self.Layers2 = nn.Sequential(
            nn.Linear(pos_embd_size * 6 + 3 + W, W),
            nn.ReLU(),
            nn.Linear(W, W),
            nn.ReLU(),
            nn.Linear(W, W),
            nn.ReLU(),
            nn.Linear(W, W + 1),
        )

        self.Layers3 = nn.Sequential(
            nn.Linear(dir_embd_size * 6 + 3 + W, int(W / 2)),
            nn.ReLU(),
        )

        self.Layers4 = nn.Sequential(
            nn.Linear(int(W / 2), 3),
            nn.Sigmoid()
        )

    def forward1(self, pos, dir):
        ''' 
        Takes: 
        pos: Position Encoding of 3D point in space
        dir: View directio or direction from which ray approaches in space
        Returns:
        colours and sigma value (volume density of point in space)
        ''' 
        output1 = self.Layers1(pos)
        input2 = torch.cat((output1, pos), dim=-1)  # SKIP CONNECTION
        output2 = self.Layers2(input2)
        sigma = nn.ReLU(output2[:, -1])
        input3 = torch.cat((output2[:, :-1], dir), dim=-1)
        output3 = self.Layers3(input3)
        colors = self.Layers4(output3)
        return colors, sigma
    


########################################## TINY NERF ##########################################

class tinyNeRF(nn.Module): # SIMPLIFIED AND SMALL MODEL
    def __init__(self, size, W): # INPUR ARGS: SIZE OF INPUT TENSOR, WIDTH OF HIDDEN LAYER
        super(tinyNeRF, self).__init__() # CALLS THE INITIALIZATION METHOD OF THE nn.Module

        self.Layers1 = nn.Sequential(
            nn.Linear(size - 6, W),
            nn.ReLU(inplace=True),
            nn.Linear(W, W),
            nn.ReLU(inplace=True),
            nn.Linear(W, W),
            nn.ReLU(inplace=True),
            nn.Linear(W, W),
            nn.ReLU(inplace=True),
        )

        self.Layers2 = nn.Sequential(
            nn.Linear(size - 6 + W, W),
            nn.ReLU(inplace=True),
            nn.Linear(W, W),
            nn.ReLU(inplace=True),
            nn.Linear(W, W),
            nn.ReLU(inplace=True),
            nn.Linear(W, 4),
        )

    def forward2(self, pos):
        ''' 
        Takes: 
        pos: Position Encoding of 3D point in space
        Returns:
        colours and sigma value (volume density of point in space)
        ''' 
        output1 = self.Layers1(pos)
        input2 = torch.cat((output1, pos), dim=-1)  # SKIP CONNECTION
        output2 = self.Layers2(input2)
        sigma = torch.relu(output2[:, -1])
        colors = torch.sigmoid(output2[:, :-1])
        return colors, sigma
    
