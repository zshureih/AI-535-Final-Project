import re
import sched
from numpy import source
import pandas as pd
import numpy as np
from random import shuffle
import os
from os import listdir, replace, times
from os.path import isfile, join, isdir
import torch
from random import shuffle
from torch.utils.data import random_split
import torchvision
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import PIL.Image as Image
from tqdm import tqdm
import json

from torch.utils.tensorboard import SummaryWriter

from sequence_model import TransformerModel_XYZRGBD, generate_square_subsequent_mask

SEQUENCE_FEATURES = ["3d_pos_x","3d_pos_y","3d_pos_z", "2d_bbox_x", "2d_bbox_y", "2d_bbox_w", "2d_bbox_h", "timestep"]
IMG_FEATURES = []
SEQUENCE_DIM = len(SEQUENCE_FEATURES)
batch_size = 12
lr = 1e-4

MASK = 99.
PAD = -99.

torch.multiprocessing.set_sharing_strategy('file_system')
writer = SummaryWriter(f"/home/zshureih/MCS/opics/output/logs/batch_{batch_size}_xyz_delta_rgbd_sequence_model_lr_{lr}")

dataset_dir = "/media/zshureih/Hybrid Drive/eval_5_dataset"
# dataset_dir = "/media/zshureih/Hybrid Drive/eval5_bug_set"
validation_dir = "/media/zshureih/Hybrid Drive/eval5_dataset_6"

# dataset_dir = os.path.join("C:", "\\Users", "Zeyad", "AI-535-Final-Project", "eval5_dataset_1")
# dataset_dir = "/nfs/hpc/share/shureihz/opics_data/eval5_dataset_1"
# validation_dir = os.path.join("C:", "\\Users", "Zeyad", "AI-535-Final-Project", "eval5_dataset_1")

# pretrained_weights = "/home/zshureih/MCS/opics/output/ckpts/13_mask_when_hidden_xyz_plus_rgbd_model_all_5.pth"

# define the features coming out of gt.txt
features = [
        "timestep",
        "obj_id",
        "shape",
        "visibility",
        "2d_bbox_x",
        "2d_bbox_y",
        "2d_bbox_w",
        "2d_bbox_h",
        "non-actor",
        "3d_pos_x",
        "3d_pos_y",
        "3d_pos_z",
        "3d_bbox_1_x",
        "3d_bbox_1_y",
        "3d_bbox_1_z",
        "3d_bbox_2_x",
        "3d_bbox_2_y",
        "3d_bbox_2_z",
        "3d_bbox_3_x",
        "3d_bbox_3_y",
        "3d_bbox_3_z",
        "3d_bbox_4_x",
        "3d_bbox_4_y",
        "3d_bbox_4_z",
        "3d_bbox_5_x",
        "3d_bbox_5_y",
        "3d_bbox_5_z",
        "3d_bbox_6_x",
        "3d_bbox_6_y",
        "3d_bbox_6_z",
        "3d_bbox_7_x",
        "3d_bbox_7_y",
        "3d_bbox_7_z",
        "3d_bbox_8_x",
        "3d_bbox_8_y",
        "3d_bbox_8_z",
        "revised_2d_bbox_x",
        "revised_2d_bbox_y",
        "revised_2d_bbox_w",
        "revised_2d_bbox_h"
    ]

def get_drop_step(scene_name, root=dataset_dir):
    root_folder = os.path.join(root, scene_name)
    
    pole_height = np.inf

    meta_files = os.listdir(os.path.join(root_folder, "Step_Output"))
    for k in range(1, len(meta_files) + 1):
        step_meta_path = os.path.join(root_folder, "Step_Output", f"step_{k:06n}.json")
        step_meta = json.load(open(step_meta_path))

        keys_list = list(step_meta['structural_object_list'].keys())
        for key in keys_list:
            if "pole_" in key:
                if step_meta['structural_object_list'][key]["position"]["y"] < pole_height:
                    pole_height = step_meta['structural_object_list'][key]["position"]["y"]
                else:
                    return k

def get_dataset(eval=False):
    master_df = pd.DataFrame([], columns=features + ['scene_name'])
    
    master_dir = dataset_dir
    if eval:
        master_dir = validation_dir
    
    print(master_dir)

    scenes = [f for f in listdir(master_dir) if "_plaus" in f]
    shuffle(scenes)
    scenes = np.array(scenes)

    # go through each scene
    for scene_name in np.unique(scenes):
        if isdir(join(master_dir, scene_name)):
            # get the ground truth tracks
            df = get_gt(scene_name, master_dir)

            # get the unique object ids
            unique_objects = np.unique(df['obj_id'].to_numpy())
            df['scene_name'] = [scene_name for i in range(df.shape[0])]
            df['drop_step'] = [-1 if "grav_" not in scene_name else get_drop_step(scene_name, master_dir) for i in range(df.shape[0])]
            actors = []
            non_actors = []
            # filter out betweeen actors and non-actors
            for id in unique_objects:
                entry_idx = np.where(df['obj_id'].to_numpy() == id)
                if df.to_numpy()[entry_idx[0][0]][8] == 0:
                    actors.append(id)
                else:
                    non_actors.append(id)
            
            if len(non_actors) != 1 or len(actors) == 0:
                scenes = scenes[scenes != scene_name]
                continue

            # save non-actor tracks to master list
            for id in non_actors:
                track_idx = np.where(df['obj_id'].to_numpy() == id)[0]
                master_df = pd.concat([master_df, df.iloc[track_idx]], axis=0)
            
            # # save actor tracks to master list
            # for id in actors:
            #     track_idx = np.where(df['obj_id'].to_numpy() == id)[0]
            #     master_df = pd.concat([master_df, df.iloc[track_idx]], axis=0)

    master_X = []
    track_lengths = []
    scene_dict = {}
    drop_step_dict = {}
    shuffle(scenes)
    # for each scene name (shuffled)
    for s, scene_name in enumerate(scenes):
        # get all entries with that row
        idx = np.where(master_df['scene_name'] == scene_name)[0]
        scene_df = master_df.iloc[idx]
        objects = scene_df['obj_id'].unique()
        # save the scene name 
        scene_dict[len(master_X)] = [scene_name]
        drop_step_dict[len(master_X)] = scene_df['drop_step'].unique()[0]

        # get each object's track
        tracks = []
        track_length = []

        for obj_id in objects:
            # get the whole track
            track_idx = np.where(scene_df['obj_id'] == obj_id)
            track = scene_df[SEQUENCE_FEATURES].iloc[track_idx].to_numpy().astype(np.float64)

            # get the packed sequence of positions
            track = torch.tensor(track).unsqueeze(0)

            # skip scenes without occlusion events, unless we're in gravity
            if len(find_gaps(track[:, :, -1].squeeze())) == 0 and "grav" not in scene_name:
                print(scene_name, "quitting")
                continue

            # add the new true trajectory to the dataset
            track_length.append(torch.max(track[:, :, -1]))
            
            # save the seqeunce without any padding
            tracks.append(track)

        if len(tracks) == 0:
            continue

        master_X.append(tracks)
        track_lengths.append(track_length)

    return master_X, track_lengths, scene_dict, drop_step_dict

def find_gaps(timesteps):
    """Generate the gaps in the list of timesteps."""
    all_steps = []

    for i in range(int(timesteps[0]), int(timesteps[-1]) + 1):
        all_steps.append(i)

    for i in range(0, len(timesteps)):
        all_steps.remove(int(timesteps[i]))
    
    return all_steps

def get_gt(scene_name, dir):
    return pd.read_csv(f"{dir}/{scene_name}/gt.txt", header=None, names=features)

def get_deltas(source):
    # go down to the absolute basics
    unpadded_idx = torch.where(source != PAD)
    unpadded = source[:, torch.unique(unpadded_idx[1]), :]
    unmasked_idx = torch.where(unpadded != MASK)
    unmasked = unpadded[:, torch.unique(unmasked_idx[1]), :]
    
    # subtract the initial position of the trajectory from each position in trajectory to delta-ize
    comp_x = torch.cat((unmasked[:, 0, :].unsqueeze(1), unmasked[:, :-1, :]), axis=1)
    deltas = torch.sub(unmasked, comp_x)

    deltas = get_norm_from_deltas(deltas)

    # replace the non MASK tokens from the unpadded sequence with the unmasked deltas
    unpadded[:, torch.unique(unmasked_idx[1]), :] = deltas
    # replace the non PAD tokens from the source sequence with the delta-ized unpadded sequence
    source[:, torch.unique(unpadded_idx[1]), :] = unpadded

    return source

def get_self_normalized(source):
    pass

def get_norm_from_deltas(src):
    # first position is always (0,0,0)
    # after that, values are added
    return torch.cumsum(src, axis=1)

class MCS_Sequence_Dataset(Dataset):
    def __init__(self, eval=False) -> None:
        super().__init__()
        
        self.eval = eval

        # get all our data
        X, L, S, D = get_dataset(eval=eval)

        self.data = {i: X[i] for i in range(len(X))}
        self.lengths = L
        self.scene_names = S
        self.drop_steps = D

        self.max_length = 300

    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def show(imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.detach()
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    def get_image_channels(self, n, bboxes, timesteps, scene_name, pad=100, root=dataset_dir):
        class ShiftToMean(object):
            def __call__(self, image):
                temp = torch.Tensor([104/255,117/255,124/255]).repeat(image.size(1),image.size(2),1).permute(2, 0, 1)
                image = image - temp
                return image

        rgb_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(
                (100, 100)
            ),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # ShiftToMean()
        ])
        depth_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(
                (100, 100)
            ),
            torchvision.transforms.Grayscale(),
            torchvision.transforms.ToTensor(),
        ])
       
        root_folder = os.path.join(root, scene_name)
        # takes 2dbbox coordinates, timesteps, 
        # and a padding value to extract 
        # windows around a target object
        bboxes = bboxes.squeeze(0)

        # start with constructing our image paths
        rgb = []
        depth = []
        for t in timesteps.int():
            rgb_path = os.path.join(root_folder, "RGB", f"{t.item():06n}.png")
            rgb_im = F.to_tensor(Image.open(rgb_path).convert('RGB')).permute(1, 2, 0)
            
            depth_path = os.path.join(root_folder, "Depth", f"{t.item():06n}.png")
            depth_im = F.to_tensor(Image.open(depth_path)).permute(1, 2, 0)
            
            # print(bboxes.shape)
            idx = torch.where(timesteps.int() == t)
            
            # get the data around the object with padding
            try:
                max_x, max_y = min((bboxes[idx, 0] + bboxes[idx, 2] + pad).item(), depth_im.shape[1] - 1), \
                    min((bboxes[idx, 1] + bboxes[idx, 3] + pad).item(), depth_im.shape[0] - 1)
                min_x, min_y = max((bboxes[idx, 0] - pad).item(), 0), \
                    max((bboxes[idx, 1] - pad).item(), 0)
            except AttributeError as e:
                print(type(depth_im))
                print(t)
                print(idx)
                print(depth_path)
                quit()

            rgb_window = rgb_transform(rgb_im[int(min_y):int(max_y) + 1, int(min_x):int(max_x) + 1, :].permute(2, 0, 1))
            depth_window = depth_transform(depth_im[int(min_y):int(max_y) + 1, int(min_x):int(max_x) + 1, :].permute(2,0,1))

            rgb.append(rgb_window)
            depth.append(depth_window)

        return rgb, depth
    
    def mask_input(self, src, timesteps, name):
        # make source timestep first
        src = src.permute(1, 0, 2)
        max_t = int(timesteps[-1])
        min_t = int(timesteps[0])

        # start with everything masked
        new_src = torch.full((max_t, 1, 3), MASK)

        # make everything before the first visible timestep padding
        new_src[:min_t - 1, :, :] = torch.full((min_t - 1, 1, 3), PAD)

        # for each visible timestep, save the xyz location
        for i, t in enumerate(timesteps):
            new_src[t.int() - 1, :, :] = src[i, :, :]

        idx_0, idx_1, idx_2 = torch.where(new_src == MASK)
        masked_idx = torch.unique(idx_0)

        # return new source as batch first
        return new_src.permute(1, 0, 2), masked_idx

    def cook_tracks(self, object_tracks, track_lengths, scene_name, drop_Step, eval=False):
        # given N object tracks (x,y,z,t) from a single scene
        max_length = max(track_lengths)
        n = len(object_tracks)

        # timesteps = np.zeros((max_length, n))
        src = torch.full((n, int(max_length), 3), PAD)

        rgb_images = []
        depth_images = []
        for i in range(n):
            track = object_tracks[i].squeeze()
            
            # get the timesteps in which the object is visible
            time = track[:, -1]
            if "grav_" in scene_name:
                indx = torch.where(time <= drop_Step)
                indx = torch.cat((indx[0], torch.full((1,), time.size(0) - 1).long()))
                time = time[indx]

            # get the bboxes of the object in which the object is visible
            bboxes = track[:, 3:-1]

            # get the channels (as tensors) around the visible object for each timestep
            if eval:
                rgb, depth = self.get_image_channels(i, bboxes, time, scene_name, root=validation_dir)
            else:
                rgb, depth = self.get_image_channels(i, bboxes, time, scene_name, root=dataset_dir)
            
            rgb_images.append(rgb)
            depth_images.append(depth)

            track = track[:, :3]
            # track = get_deltas(track.unsqueeze(0))
            track, masked_idx = self.mask_input(track.unsqueeze(0), time, scene_name)
            src[i, :track.shape[1], :] = track

        rgb_images = [img.unsqueeze(0) for img in rgb_images[0]]
        rgb_images = torch.cat(rgb_images)

        depth_images = [img.unsqueeze(0) for img in depth_images[0]]
        depth_images = torch.cat(depth_images)

        return src, time, rgb_images, depth_images, masked_idx

    def get_masked_data(self, src, timesteps, scene_name, eval=False):
        if eval:
            root = validation_dir
        else:
            root = dataset_dir

        root_folder = os.path.join(root, scene_name)

        target = src.detach().clone()

        # plausible scenes only, so all timesteps before timesteps[-1] should have step_output data
        for t in range(timesteps[0].long(), src.size(1)):
            # if object not visible, replace src at this timestep
            if t not in timesteps.long():
                # get the xyz of the object at the timestep
                step_meta_path = os.path.join(root_folder, "Step_Output", f"step_{t:06n}.json")
                step_meta = json.load(open(step_meta_path))

                k = list(step_meta['object_list'].keys())
                hidden_xyz = list(step_meta['object_list'][k[0]]['position'].values())
                target[:, t - 1, :] = torch.Tensor(hidden_xyz)
        
        return target

    def __getitem__(self, index):
        scene = self.data[index]
        track_lengths = self.lengths[index]
        scene_name = self.scene_names[index]
        drop_step = self.drop_steps[index]

        src, timesteps, rgb, depth, masked_ = self.cook_tracks(scene, track_lengths, scene_name[0], drop_step, self.eval)

        truth = self.get_masked_data(src, timesteps, scene_name[0], self.eval)
        
        input_image = torch.cat([rgb, depth], axis=1)

        length = Tensor([src.size(1), timesteps.size(0)])

        # right side padding of source seqeunce
        seq = torch.full((self.max_length, 3), PAD)
        seq[:src.size(1), :] = src.squeeze()
        seq = get_deltas(seq.unsqueeze(0)).squeeze(0)

        # right side padding of target
        target = torch.full((self.max_length, 3), PAD)
        target[:truth.size(1), :] = truth.squeeze()
        target = get_deltas(target.unsqueeze(0)).squeeze(0)
        
        time = torch.full((self.max_length, 1), PAD)
        time[:timesteps.size(0)] = timesteps.unsqueeze(1)

        images = torch.zeros((self.max_length, 4, 100, 100))
        images[:timesteps.size(0)] = input_image

        masked_idx = torch.full((self.max_length, 1), PAD)
        masked_idx[:len(masked_)] = masked_.unsqueeze(1)

        return seq, time, images, length, target, scene_name[0], masked_idx

def eval(model, val_set, export_flag=False):
    model.eval()
    losses = []
    outputs = []
    
    with torch.no_grad():
        for i, output in enumerate(tqdm(val_set)):
            src, timesteps, input_images, length, target, scene_name, masked_idx = output[0], output[1], output[2], output[3], output[4], output[5], output[6]

            # print(src[masked_idx[length[:, 1].long()].long()].shape)
            output = model(src, timesteps, input_images, length)

            # outputs.append([src.detach().squeeze(), output.detach().squeeze(), target.detach().squeeze(), scene_name, timesteps])
            
            loss = 0
            for j in range(length.size(0)):
                # get masked idx of output        
                idx = torch.where(src[j] != PAD)
                idx = torch.unique(idx[0])

                if len(idx) == 0:
                    continue

                loss += criterion(output[j, idx].permute(1, 0, 2), target[j, idx].unsqueeze(0).cuda())

            losses.append(loss.mean().detach().item())

            writer.add_scalar("Loss/val", loss.mean().item(), (epoch * len(val_set)) + i)

    print(f"epoch {epoch:3d} - avg val loss={np.mean(losses)} - lr = {lr}")

    return losses


def train(model, train_set, epoch=0):
    model.train()
    losses = []

    for i, output in enumerate(tqdm(train_set)):
        optimizer.zero_grad()
        src, timesteps, input_images, length, target, scene_name, masked_idx = output[0], output[1], output[2], output[3], output[4], output[5], output[6]

        output = model(src, timesteps, input_images, length)
        
        # calc loss
        loss = 0
        for j in range(length.size(0)):
            # get masked idx of output        
            idx = torch.where(src[j] != PAD)
            idx = torch.unique(idx[0])
            if len(idx) == 0:
                continue

            loss += criterion(output[j, idx].permute(1, 0, 2), target[j, idx].unsqueeze(0).cuda())

        loss.mean().backward()
        
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("Loss/train", loss.mean(), (epoch * len(train_set)) + i)
        losses.append(loss.detach().item())
        print(f"epoch {epoch:3d} - batch {i:5d}/{len(train_set)} - loss={loss.mean()} - lr = {lr}")

    return losses

if __name__ == "__main__":
    # Parameters
    params = {'batch_size': batch_size,
            'shuffle': True,
            'num_workers': 1,
            'pin_memory': True
            }
    max_epochs = 20

    # grab the dataset
    full_dataset = MCS_Sequence_Dataset()

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, val_size])

    train_generator = torch.utils.data.DataLoader(train_dataset, **params)
    test_generator = torch.utils.data.DataLoader(test_dataset, **params)

    # okay let's start training
    # img_enc_dim = 128
    img_enc_dim = 256**2
    model = TransformerModel_XYZRGBD(img_enc_dim + 3, 128, 8, 128, 2, 0.2).cuda()
    
    # if pretrained_weights:
    #     model.load_state_dict(torch.load(pretrained_weights))
    #     print("pretrained weights")

    # TODO: try huber loss
    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # variables to save best model
    best_val_loss = np.inf
    best_model = None
    best_errors = ''
    best_confusion = []
    best_epoch = None

    avg_train_losses = []
    avg_val_losses = []
    val_outputs = []
    train_outputs = []
    for epoch in range(max_epochs):
        print("epoch:", epoch)

        train_losses = train(model, train_generator, epoch)
        val_losses = eval(model, test_generator, epoch)
        # scheduler.step()

        if np.mean(val_losses, axis=-1) < best_val_loss:
            best_val_loss = np.mean(val_losses)
            best_model = model.state_dict()
            best_epoch = epoch
            # save model weights
            torch.save(best_model, f"./{epoch}_batch_{batch_size}_xyz_delta_rgbd_lr_{lr}_sequence_mask_model.pth")
            # val_outputs.append(v_o)

        avg_train_losses.append(np.mean(train_losses))
        avg_val_losses.append(np.mean(val_losses))


    fig = plt.figure(figsize=(12, 4))

    ax = fig.add_subplot(121)
    ax.set_title('Average Loss per Epoch')
    ax.plot(range(len(avg_train_losses)), avg_train_losses, label='Training Loss')
    ax.plot(range(len(avg_val_losses)), avg_val_losses, label='Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('MSE Loss')
    ax.legend()

    # ax = fig.add_subplot(122, projection='3d')
    # val_outputs.reverse()
    # for info in val_outputs:
    #     src, output, target, name, time = info[0][0], info[0][1], info[0][2], info[0][3], info[0][4]
    #     for j in range(src.size(0)):
    #         ax.set_title(name[j])
    #         # print("name", name)
    #         # print("length", length)
    #         # print(output)


    #         unpad_idx = torch.where(src[j] != PAD)
    #         unpad_idx = torch.unique(unpad_idx[0])
    #         unpadded_src = src[j, unpad_idx]
    #         unpadded_output = output[j, unpad_idx].unsqueeze(1)
    #         unpadded_target = target[j, unpad_idx].unsqueeze(1)
            
    #         unmasked_idx = torch.where(unpadded_src != MASK)
    #         unmasked_idx = torch.unique(unmasked_idx[0])
    #         unmasked_src = unpadded_src[unmasked_idx].unsqueeze(1)
            
    #         # target
    #         # unpadded_target = get_norm_from_deltas(unpadded_target)
    #         print("unpadded_target.shape")
    #         print(unpadded_target)
    #         x = unpadded_target[:, :, 0].reshape(-1).cpu().numpy()
    #         z = unpadded_target[:, :, 1].reshape(-1).cpu().numpy()
    #         y = unpadded_target[:, :, 2].reshape(-1).cpu().numpy()
    #         ax.scatter(x,y,z,c='green', label="target trajectory")

    #         # unmasked_src = get_norm_from_deltas(unmasked_src)
    #         print("unmasked_src.shape")
    #         print(unmasked_src)
    #         x = unmasked_src[:, :, 0].reshape(-1).cpu().numpy()
    #         z = unmasked_src[:, :, 1].reshape(-1).cpu().numpy()
    #         y = unmasked_src[:, :, 2].reshape(-1).cpu().numpy()
    #         ax.scatter(x,y,z,c='blue',label="source trajectory")

    #         # unpadded_output = get_norm_from_deltas(unpadded_output)
    #         print("unpadded_output.shape")
    #         print(unpadded_output.shape)
    #         print(unpadded_output)
    #         x = unpadded_output[:, :, 0].reshape(-1).cpu().numpy()
    #         z = unpadded_output[:, :, 1].reshape(-1).cpu().numpy()
    #         y = unpadded_output[:, :, 2].reshape(-1).cpu().numpy()
    #         ax.scatter(x,y,z,c='red',label="output trajectory")

    #         break
        
    #     break

    # ax.set_xlabel('X')
    # ax.set_ylabel('Z')
    # ax.set_zlabel('Y')
    # ax.legend()

    plt.show(block=True)