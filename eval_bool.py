from scripts.pvoe.transformer.train_scene_xyz_rgbd import MCS_Sequence_Dataset
from opics.pvoe.transformer.bool_models import TransformerModel_XYZRGBD, generate_square_subsequent_mask
import numpy as np
import torch
from torch import nn, Tensor
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
    img_enc_dim = 256**2
    model = TransformerModel_XYZRGBD(img_enc_dim + 3, 128, 8, 128, 2, 0.2).cuda()
    model.load_state_dict(torch.load("/home/zshureih/MCS/opics/output/ckpts/13_mask_when_hidden_xyz_plus_rgbd_model_all_5.pth"))
    criterion = nn.BCELoss()

    def eval(model, val_set, export_flag=False):
        model.eval()
        losses = []
        outputs = []
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
                
                # save losses, as well as scene specifics 
                losses.append(loss.item())
                outputs.append((scene_name[0], plausibility.detach().item(), output.detach().item()))

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
                print(f"running accuracy {total_correct}/{i+1}={total_correct / (i+1):04f}")

        print("val-accuracy:", total_correct / len(val_set))
        confusion_matrix = [ [plaus_correct / (plaus_correct + plaus_incorrect), plaus_incorrect / (plaus_correct + plaus_incorrect)],
                            [implaus_incorrect / (implaus_correct + implaus_incorrect), implaus_correct / (implaus_correct + implaus_incorrect)]
                        ]

        return total_correct / len(val_set), losses, incorrect_scenes, confusion_matrix

    accuracy, losses, _scenes, _m_conf = eval(model, full_generator)

    fig, ax1 = plt.subplots(1, 1, sharex=True)
    cax = ax1.matshow(_m_conf)
    fig.colorbar(cax)
    ax1.set_xticks(np.arange(2))
    ax1.set_yticks(np.arange(2))
    ax1.set_xticklabels(["Predicted Plausible", "Predicted Implausible"])
    ax1.set_yticklabels(["Plausible Scene", "Implausible Scene"])

    plt.show(block=True)

    with open(f"home/zshureih/MCS/opics/output/ckpts/eval_5_mask_when_hidden_xyz_plus_rgbd_model.txt", "w+") as outfile:
        for scene_name in _scenes:
            outfile.write(scene_name[0] + "\n")
