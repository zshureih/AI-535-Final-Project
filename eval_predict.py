from operator import mod
from predict import MCS_Sequence_Dataset
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
# from sequence_model import TransformerModel, generate_square_subsequent_mask
import matplotlib.pyplot as plt

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
    
if __name__ == "__main__":
    params = {'batch_size': 1,
            'shuffle': True,
            'num_workers': 3}

    full_dataset = MCS_Sequence_Dataset()
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, val_size])

    train_generator = torch.utils.data.DataLoader(train_dataset, **params)
    test_generator = torch.utils.data.DataLoader(test_dataset, **params)

    # okay let's start evaluating
    lr = 0.00001  # learning rate
    model = TransformerModel(3, 128, 8, 128, 2, 0.2).cuda()
    model.load_state_dict(torch.load("100_delta_model.pth"))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.HuberLoss()

    def eval(model, val_set):
        model.eval()
        losses = []
        outputs = []

        best_loss = np.inf

        with torch.no_grad():
            for src, timesteps, scene_names, lengths in val_set:
                optimizer.zero_grad()

                src = get_deltas(src) # get our deltas
                # src, target, mask_idx = mask_input(src, timesteps) # mask some of the inputs
                src_mask = generate_square_subsequent_mask(src.size(0)).cuda()
                
                output = model(src, src_mask)
                total_loss = criterion(output, src)

                outputs.append([src, output, scene_names, lengths])

                losses.append([total_loss.item()])

        return outputs, losses

    outputs, losses = eval(model, test_generator)

    for out in zip(outputs, losses):
        fig = plt.figure(figsize=(6, 4))
        
        info = out[0]
        loss = out[1]
        src, output, name, length = info[0], info[1], info[2], info[3]

        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f"{name} - Total Track Loss={loss[0]:04f}")

        # x = target[:, :, 0].reshape(-1).cpu().numpy()
        # y = target[:, :, 1].reshape(-1).cpu().numpy()
        # z = target[:, :, 2].reshape(-1).cpu().numpy()
        # ax.scatter(x,y,z,c='red', label="source trajectory")

        src = get_norm_from_deltas(src)
        x = src[:, :, 0].reshape(-1).cpu().numpy()
        y = src[:, :, 1].reshape(-1).cpu().numpy()
        z = src[:, :, 2].reshape(-1).cpu().numpy()
        ax.scatter(x,y,z,c='blue',label="target trajectory")

        output = get_norm_from_deltas(output)
        x = output[:, :, 0].reshape(-1).cpu().numpy()
        y = output[:, :, 1].reshape(-1).cpu().numpy()
        z = output[:, :, 2].reshape(-1).cpu().numpy()
        ax.scatter(x,y,z,c='green',label="output trajectory")

        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        ax.set_xticks(range(-10, 10))
        ax.set_yticks(range(-10, 10))
        ax.set_zticks(range(-10, 1))
        ax.legend()

        plt.show(block=True)