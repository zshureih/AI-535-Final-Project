import re
import sched
import pandas as pd
import numpy as np
from random import shuffle
import os
from os import listdir, replace
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

from bool_models import TransformerModel_XYZRGBD, generate_square_subsequent_mask

torch.multiprocessing.set_sharing_strategy('file_system')
writer = SummaryWriter("/home/zshureih/MCS/opics/output/logs/xyz_rgbd_model_lr_1e-5")

dataset_dir = "/media/zshureih/Hybrid Drive/eval_5_dataset"
validation_dir = "/media/zshureih/Hybrid Drive/eval5_dataset_6"
# dataset_dir = "/media/zshureih/Hybrid Drive/eval5_dataset_6"

SEQUENCE_FEATURES = ["3d_pos_x","3d_pos_y","3d_pos_z", "2d_bbox_x", "2d_bbox_y", "2d_bbox_w", "2d_bbox_h", "timestep"]
IMG_FEATURES = []
SEQUENCE_DIM = len(SEQUENCE_FEATURES)

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

def get_dataset(eval=False):
    master_df = pd.DataFrame([], columns=features + ['scene_name'])
    
    master_dir = dataset_dir
    if eval:
        master_dir = validation_dir
    
    print(master_dir)

    scenes = listdir(master_dir)
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
            actors = []
            non_actors = []
            # filter out betweeen actors and non-actors
            for id in unique_objects:
                entry_idx = np.where(df['obj_id'].to_numpy() == id)
                if df.to_numpy()[entry_idx[0][0]][8] == 0:
                    actors.append(id)
                else:
                    non_actors.append(id)

            # if len(non_actors) == 2 and "coll_" in scene_name and "_plaus" in scene_name:
            #     # flip a coin and remove a non_actor
            #     flip = np.random.randint(0, 2)
            #     non_actors.remove(non_actors[flip])
            
            if len(non_actors) != 1:
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
    shuffle(scenes)
    # for each scene name (shuffled)
    for s, scene_name in enumerate(scenes):
        # get all entries with that row
        idx = np.where(master_df['scene_name'] == scene_name)[0]
        scene_df = master_df.iloc[idx]
        objects = scene_df['obj_id'].unique()
        # save the scene name 
        scene_dict[len(master_X)] = [scene_name]

        # get each object's track
        tracks = []
        track_length = []

        for obj_id in objects:
            # get the whole track
            track_idx = np.where(scene_df['obj_id'] == obj_id)
            track = scene_df[SEQUENCE_FEATURES].iloc[track_idx].to_numpy().astype(np.float64)

            # get the packed sequence of positions
            track = torch.tensor(track).unsqueeze(0)    
            # add the new true trajectory to the dataset
            track_length.append(torch.max(track[:, :, -1]))
            
            # save the seqeunce without any padding
            tracks.append(track)

        master_X.append(tracks)
        track_lengths.append(track_length)

    return master_X, track_lengths, scene_dict

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
    # subtract the initial position of the trajectory from each position in trajectory to delta-ize
    comp_x = torch.cat((source[:, 0, :].unsqueeze(1), source[:, :-1, :]), axis=1)
    deltas = torch.sub(source, comp_x)
    return deltas

def get_self_normalized(source):
    pass

def get_norm_from_deltas(src):
    # first position is always (0,0,0)
    # after that, values are added
    return torch.cumsum(src, axis=0)

class MCS_Sequence_Dataset(Dataset):
    def __init__(self, eval=False) -> None:
        super().__init__()
        
        self.eval = eval

        # get all our data
        X, L, S = get_dataset(eval=eval)

        self.data = {i: X[i] for i in range(len(X))}
        self.lengths = L
        self.scene_names = S

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

    def get_image_channels(self, n, bboxes, timesteps, scene_name, pad=75, root=dataset_dir):
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
            # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
    
    def mask_input(self, src, timesteps):
        # make source timestep first
        src = src.permute(1, 0, 2)
        max_t = int(timesteps[-1])

        new_src = torch.full((max_t, 1, 3), 99.)

        for i, t in enumerate(timesteps):
            new_src[t.int() - 1, :, :] = src[i, :, :]

        # return new source as batch first
        return new_src.permute(1, 0, 2)

    def cook_tracks(self, object_tracks, track_lengths, scene_name, eval=False):
        # given N object tracks (x,y,z,t) from a single scene
        max_length = max(track_lengths)
        n = len(object_tracks)

        # timesteps = np.zeros((max_length, n))
        src = torch.full((n, int(max_length), 3), 99.)

        rgb_images = []
        depth_images = []
        for i in range(n):
            track = object_tracks[i].squeeze()
            
            # get the timesteps in which the object is visible
            time = track[:, -1]
            # for t in time.view(-1):
                # timesteps[t.int(), i] = t

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
            track = self.mask_input(track.unsqueeze(0), time)
            
            src[i, :track.shape[1], :] = track

        rgb_images = [img.unsqueeze(0) for img in rgb_images[0]]
        rgb_images = torch.cat(rgb_images)

        depth_images = [img.unsqueeze(0) for img in depth_images[0]]
        depth_images = torch.cat(depth_images)

        return src, time, rgb_images, depth_images

    def __getitem__(self, index):
        # get the scene
        scene = self.data[index]
        track_lengths = self.lengths[index]
        scene_name = self.scene_names[index]

        src, timesteps, rgb, depth = self.cook_tracks(scene, track_lengths, scene_name[0], self.eval)

        input_image = torch.cat([rgb, depth], axis=1)

        return src, timesteps, input_image, scene_name[0]


def eval(model, val_set, export_flag=False):
    model.eval()
    losses = []
    incorrect_scenes = []
    total_correct = 0
    
    plaus_correct = 0
    plaus_incorrect = 0

    implaus_correct = 0
    implaus_incorrect = 0

    with torch.no_grad():
        for i, output in enumerate(tqdm(val_set)):
            src, timesteps, input_images, scene_name = output[0], output[1], output[2], output[3]
            
            plausibility = Tensor([1]) if "_plaus" in scene_name[0] else Tensor([0])
            output = model(src.cuda(), timesteps, input_images)
            
            loss = criterion(output.squeeze(0), plausibility.cuda())

            losses.append(loss.detach().item())

            if output < 0.5:
                output = 0
            else:
                output = 1
            
            if output == plausibility.item():
                if plausibility.item() == 1:
                    plaus_correct += 1
                else:
                    implaus_correct += 1
                total_correct += 1
            else:
                incorrect_scenes.append(scene_name)
                if plausibility.item() == 1:
                    plaus_incorrect += 1
                else:
                    implaus_incorrect += 1
            
            if i == 0:
                grid = torchvision.utils.make_grid(input_images[0, :, :3, :, :])
                writer.add_image(f'Val Scene {scene_name} {plausibility}-{output}', grid)

            writer.add_scalar("Loss/val", loss.item(), (epoch * len(val_set)) + i)

    print("val-accuracy:", total_correct / len(val_set))

    confusion_matrix = [ [plaus_correct / (plaus_correct + plaus_incorrect), plaus_incorrect / (plaus_correct + plaus_incorrect)],
                         [implaus_incorrect / (implaus_correct + implaus_incorrect), implaus_correct / (implaus_correct + implaus_incorrect)]
                       ]

    return total_correct / len(val_set), losses, incorrect_scenes, confusion_matrix


def train(model, train_set, epoch=0):
    model.train()
    total_loss = 0
    log_interval = 50
    losses = []
    total_correct = 0
    
    plaus_correct = 0
    plaus_incorrect = 0

    implaus_correct = 0
    implaus_incorrect = 0

    # online method for now?
    for i, output in enumerate(tqdm(train_set)):
        optimizer.zero_grad()
        src, timesteps, input_images, scene_name = output[0], output[1], output[2], output[3]
        # print(src.shape)
        # print(timesteps.shape)
        # print(input_images.shape)
        # print(scene_name)

        plausibility = Tensor([1]) if "_plaus" in scene_name[0] else Tensor([0])

        output = model(src.cuda(), timesteps, input_images)

        loss = criterion(output.squeeze(0), plausibility.cuda())
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        total_loss += loss.detach().item()

        if output < 0.5:
            rating = 0
        else:
            rating = 1

        if rating == plausibility.item():
            if plausibility.item() == 1:
                plaus_correct += 1
            else:
                implaus_correct += 1
            total_correct += 1
        else:
            if plausibility.item() == 1:
                plaus_incorrect += 1
            else:
                implaus_incorrect += 1

        if i % log_interval == 0 and i > 0:
            lr = optimizer.param_groups[0]['lr']
            cur_loss = total_loss / log_interval
            writer.add_scalar("Loss/train", cur_loss,( epoch * len(train_set)) + i)
            losses.append(cur_loss)
            # print(f"epoch {epoch:3d} - batch {i:5d}/{len(train_set)} - loss={cur_loss} - lr = {lr}")
            total_loss = 0
    
    print("train-accuracy:", total_correct / len(train_set))

    return losses, total_correct / len(train_set)

if __name__ == "__main__":
    # Parameters
    params = {'batch_size': 1,
            'shuffle': True,
            'num_workers': 4,
            }
    max_epochs = 100

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
    
    # TODO: try huber loss
    criterion = nn.BCELoss()

    lr = 1e-5  # learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # variables to save best model
    best_val_acc = 0
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

        train_losses, train_accuracies = train(model, train_generator, epoch)
        writer.add_scalar("Acc/train", train_accuracies, epoch)
        val_accuracies, val_losses, incorrect, confusion = eval(model, test_generator, epoch)
        writer.add_scalar("Acc/val", val_accuracies, epoch)
        # scheduler.step()

        if val_accuracies > best_val_acc:
            best_val_acc = val_accuracies
            best_model = model.state_dict()
            best_errors = incorrect
            best_confusion = confusion
            best_epoch = epoch
            # save model weights
            torch.save(best_model, f"./{epoch}_mask_when_hidden_xyz_rgbd_model_all_5.pth")

        avg_train_losses.append(np.mean(train_losses))
        avg_val_losses.append(np.mean(val_losses))
        val_outputs.append(val_accuracies)
        train_outputs.append(train_accuracies)


    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.set_title('Average Loss per Epoch')
    ax1.plot(range(len(avg_train_losses)), avg_train_losses, label='Training Loss')
    ax1.plot(range(len(avg_val_losses)), avg_val_losses, label='Validation Loss')
    ax1.set_xticks(np.arange(len(avg_train_losses)))
    ax1.set_xticklabels(np.arange(len(avg_train_losses)))
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('BCE Loss')
    ax1.legend()

    ax2.set_title(f'Accuracy on Validation set ({len(test_generator)} samples)')
    ax2.plot(range(len(train_outputs)), train_outputs, label='Training Accuracy')
    ax2.plot(range(len(val_outputs)), val_outputs, label="Validation Accuracy")
    ax2.set_xticks(np.arange(len(val_outputs)))
    ax2.set_xticklabels(np.arange(len(avg_train_losses)))
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')

    plt.show(block=True)

    fig, ax1 = plt.subplots(1, 1, sharex=True)
    cax = ax1.matshow(best_confusion)
    fig.colorbar(cax)
    ax1.set_xticks(np.arange(2))
    ax1.set_yticks(np.arange(2))
    ax1.set_xticklabels(["Predicted Plausible", "Predicted Implausible"])
    ax1.set_yticklabels(["Plausible Scene", "Implausible Scene"])

    plt.show(block=True)

    with open(f"{best_epoch}_mask_when_hidden_xyz_rgbd_model_all_5.txt", "w+") as outfile:
        for scene_name in best_errors:
            outfile.write(scene_name[0] + "\n")
