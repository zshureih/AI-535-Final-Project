from train_scene_classification_xyz_rgbd import MCS_Sequence_Dataset
from bool_models import TransformerModel_XYZRGBD, generate_square_subsequent_mask
import numpy as np
import torchvision
import torch
from torch import nn, Tensor
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

MASK = 99.
PAD = -99.

if __name__ == "__main__":
    params = {'batch_size': 1,
            'shuffle': True,
            'num_workers': 0}

    full_dataset = MCS_Sequence_Dataset(eval=True)
    full_generator = torch.utils.data.DataLoader(full_dataset, **params)

    print(len(full_dataset), "trajectories")

    # okay let's start evaluating
    lr = 5e-4  # learning rate
    # img_enc_dim = 64
    img_enc_dim = 512
    model = TransformerModel_XYZRGBD(img_enc_dim + 3, 128, 8, 128, 2, 0.2).cuda()
    model.init_classification_head()
    model.load_state_dict(torch.load("/home/zshureih/MCS/opics/output/ckpts/1_batch_14_xyz_rgbd_classifier_lr_0.0001.pth"))
    criterion = nn.BCELoss()

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
                src, timesteps, input_images, length, plausibility, scene_name = output[0], output[1], output[2], output[3], output[4], output[5]
            
                output = model.get_per_step_binaries(src, input_images, length)

                # for each scene create a graph of the outputs with plt
                plt.plot(range(len(output[0])), list(output[0].squeeze().detach().cpu()), label='softmax at each timestep')
                plt.title(f'{scene_name}')
                plt.xlabel('Time')
                plt.ylabel('Softmax Output')
                plt.savefig(f"./eval_each_step/graphs/{scene_name[0]}")
                plt.close()

                # get a list of frames from the input sequence
                frames = torch.zeros((len(output[0]), 100, 100, 3))
                x = src[0, :].unsqueeze(0).cuda()
                unpadded_idx = torch.where(x != PAD)
                unpadded_idx = torch.unique(unpadded_idx[1])
                unpadded_x = src[:, unpadded_idx, :].permute(1, 0, 2)
                unmmask_idx = torch.where(unpadded_x != MASK)
                unmmask_idx = torch.unique(unmmask_idx[0])
                images = input_images[0, :length[0, 1].long()]

                frames[unmmask_idx] = images.permute(0, 2, 3, 1)[:, :, :, :-1]
                for j, img in enumerate(frames):
                    plt.imshow(F.to_pil_image(img.permute(2, 0, 1)))
                    plt.text(0, 10, f"{output[0][j].squeeze().detach().cpu().item():03f}", bbox=dict(fill=True, edgecolor='red', linewidth=2), color='red')
                    if not os.path.exists(f"./eval_each_step/images/{scene_name[0]}"):
                        os.mkdir(f"./eval_each_step/images/{scene_name[0]}")
                    plt.savefig(f"./eval_each_step/images/{scene_name[0]}/{j}.png")
                    plt.close()


                if output[0][-1].squeeze().detach().cpu().item() < 0.5:
                    rating = 0
                else:
                    rating = 1

                if rating == plausibility[0].item():
                    if plausibility[0].item() == 1:
                        plaus_correct += 1
                    else:
                        implaus_correct += 1
                    total_correct += 1
                else:
                    incorrect_scenes.append(scene_name)
                    if plausibility[0].item() == 1:
                        plaus_incorrect += 1
                    else:
                        implaus_incorrect += 1

        print(f"{plaus_correct + plaus_incorrect}:{implaus_correct + implaus_incorrect}")
        print("val-accuracy:", total_correct / (len(val_set)))

        confusion_matrix = [ [plaus_correct / (plaus_correct + plaus_incorrect), plaus_incorrect / (plaus_correct + plaus_incorrect)],
                            [implaus_incorrect / (implaus_correct + implaus_incorrect), implaus_correct / (implaus_correct + implaus_incorrect)]
                        ]

        return total_correct / (len(val_set)), losses, incorrect_scenes, confusion_matrix


    accuracy, losses, _scenes, _m_conf = eval(model, full_generator)

    fig, ax1 = plt.subplots(1, 1, sharex=True)
    cax = ax1.matshow(_m_conf)
    fig.colorbar(cax)
    ax1.set_xticks(np.arange(2))
    ax1.set_yticks(np.arange(2))
    ax1.set_xticklabels(["Predicted Plausible", "Predicted Implausible"])
    ax1.set_yticklabels(["Plausible Scene", "Implausible Scene"])

    plt.show(block=True)

    with open(f"/home/zshureih/MCS/opics/output/ckpts/eval_5_mask_when_hidden_xyz_plus_rgbd_model_v2.txt", "w+") as outfile:
        for scene_name in _scenes:
            outfile.write(scene_name[0] + "\n")
