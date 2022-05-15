from operator import mod

from numpy import block
from train_mask_prediction_xyz_rgbd import MCS_Sequence_Dataset
import re
import pandas as pd
import numpy as np
import os
from os import listdir, replace
from os.path import isfile, join, isdir
import torch
from random import shuffle
from torch.utils.data import random_split
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.utils.data.dataset import Dataset
from sequence_model import TransformerModel_XYZRGBD, generate_square_subsequent_mask
import matplotlib.pyplot as plt
from tqdm import tqdm

MASK = 99.
PAD = -99.

def mask_input(src, timesteps, min_k=5):
    src = src.permute(1, 0, 2).cuda()
    target = src.detach().clone() # make a copy of our source and label it as the target

    # now let's mask the input
    # randomly select a k frame window to mask 
    init_index = np.random.randint(0, src.size(0) - min_k)
    masked_idx = range(init_index, init_index + min_k)

    # mask idx 
    for t in masked_idx:
        src[t, :, :] = torch.full((1, 3), -99, dtype=torch.float64).cuda()

    return src, target, masked_idx

def get_deltas(source):
    # subtract the initial position of the trajectory from each position in trajectory to delta-ize
    comp_x = torch.cat((source[:, 0, :].unsqueeze(1), source[:, :-1, :]), axis=1)
    deltas = torch.sub(source, comp_x)
    return deltas.cuda()

def get_self_normalized(source):
    pass

def get_norm_from_deltas(src):
    # first position is always (0,0,0)
    # after that, values are added
    return torch.cumsum(src, axis=1)

def save_traj_figure(src, target, output, name, loss):
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"{name} - Total Track Loss={loss.item():04f}")

    unpad_idx = torch.where(src != PAD)
    unpad_idx = torch.unique(unpad_idx[0])
    unpadded_src = src[unpad_idx]
    unpadded_output = output[unpad_idx]
    unpadded_target = target[unpad_idx].unsqueeze(1)
    
    unmasked_idx = torch.where(unpadded_src != MASK)
    unmasked_idx = torch.unique(unmasked_idx[0])
    unmasked_src = unpadded_src[unmasked_idx].unsqueeze(1)
    
    # target
    # unpadded_target = get_norm_from_deltas(unpadded_target)
    x = unpadded_target[:, :, 0].reshape(-1).cpu().numpy()
    z = unpadded_target[:, :, 1].reshape(-1).cpu().numpy()
    y = unpadded_target[:, :, 2].reshape(-1).cpu().numpy()
    ax.plot(x,y,z,c='green', label="target trajectory")
    ax.scatter(x,y,z,c='green', label="target trajectory")

    # unmasked_src = get_norm_from_deltas(unmasked_src)
    x = unmasked_src[:, :, 0].reshape(-1).cpu().numpy()
    z = unmasked_src[:, :, 1].reshape(-1).cpu().numpy()
    y = unmasked_src[:, :, 2].reshape(-1).cpu().numpy()
    ax.plot(x,y,z,c='blue',label="source trajectory")
    ax.scatter(x,y,z,c='blue',label="source trajectory")

    # unpadded_output = get_norm_from_deltas(unpadded_output)
    x = unpadded_output[:, :, 0].reshape(-1).cpu().numpy()
    z = unpadded_output[:, :, 1].reshape(-1).cpu().numpy()
    y = unpadded_output[:, :, 2].reshape(-1).cpu().numpy()
    ax.plot(x,y,z,c='red',label="output trajectory")
    ax.scatter(x,y,z,c='red',label="output trajectory")

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.legend()

    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_zlim([-10, 10])
    # plt.show(block=True)
    plt.savefig(f"{name}-{loss.item()}.jpg")
    plt.close()

    
if __name__ == "__main__":
    params = {'batch_size': 3,
            'shuffle': True,
            'num_workers': 1}

    full_dataset = MCS_Sequence_Dataset(eval=True)
    # train_size = int(0.8 * len(full_dataset))
    # val_size = len(full_dataset) - train_size
    # train_dataset, test_dataset = random_split(full_dataset, [train_size, val_size])

    full_generator = torch.utils.data.DataLoader(full_dataset, **params)

    # okay let's start evaluating
    lr = 1e-3  # learning rate

    img_enc_dim = 256**2
    model = TransformerModel_XYZRGBD(img_enc_dim + 3, 128, 8, 128, 2, 0.2).cuda()
    model.load_state_dict(torch.load("4_batch_12_xyz_delta_rgbd_lr_0.001_sequence_mask_model.pth"))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    def eval(model, val_set, export_flag=False):
        model.eval()
        losses = []
        
        with torch.no_grad():
            for i, output in enumerate(tqdm(val_set)):
                src, timesteps, input_images, length, target, scene_name, masked_idx = output[0], output[1], output[2], output[3], output[4], output[5], output[6]

                # print(src[masked_idx[length[:, 1].long()].long()].shape)
                output = model(src, timesteps, input_images, length)
                
                loss = 0
                for j in range(length.size(0)):
                    # get masked idx of output        
                    idx = torch.where(src[j] == MASK)
                    idx = torch.unique(idx[0])

                    if len(idx) == 0:
                        continue

                    l = criterion(output[j, idx].permute(1, 0, 2), target[j, idx].unsqueeze(0).cuda())

                    # visualize the plot the trajectories and save the output
                    save_traj_figure(src[j], target[j], output[j], scene_name[j], l)

                    loss += l
                
                losses.append(loss.mean().detach().item())

        return losses

    losses = eval(model, full_generator)