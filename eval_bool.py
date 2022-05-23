from train_scene_classification_xyz_rgbd import MCS_Sequence_Dataset
from bool_models import TransformerModel_XYZRGBD, generate_square_subsequent_mask
import numpy as np
import torchvision
import torch
from torch import nn, Tensor
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm


if __name__ == "__main__":
    params = {'batch_size': 1,
            'shuffle': True,
            'num_workers': 0}

    full_dataset = MCS_Sequence_Dataset(eval=True)
    full_generator = torch.utils.data.DataLoader(full_dataset, **params)

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
            
                output = model(src, timesteps, input_images, length)
                loss = criterion(output.squeeze(-1), plausibility.cuda())

                losses.append(loss.detach().cpu().item())

                for j in range(len(output)):
                    if output[j].detach().cpu().item() < 0.5:
                        rating = 0
                    else:
                        rating = 1

                    if rating == plausibility[j].item():
                        if plausibility[j].item() == 1:
                            plaus_correct += 1
                        else:
                            implaus_correct += 1
                        total_correct += 1
                    else:
                        incorrect_scenes.append(scene_name)
                        if plausibility[j].item() == 1:
                            plaus_incorrect += 1
                        else:
                            implaus_incorrect += 1
                
                    if i == 0:
                        grid = torchvision.utils.make_grid(input_images[0, j, :3, :, :])



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
