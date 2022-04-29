import math
from typing import Tuple
import numpy as np

import torch
from torch import nn, Tensor
import torchvision.transforms.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

import matplotlib.pyplot as plt

class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride=1) -> None:
        super().__init__()

        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            self.skip.apply(self.init_cnn_weights)
        else:
          self.skip = None

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels))

        self.block.apply(self.init_cnn_weights)

        self.relu = nn.ReLU()

    def init_cnn_weights(self, m) -> None:
        if isinstance(m, nn.Conv2d):
            n = m.in_channels
            y = 1.0/np.sqrt(n)
            m.weight.data.uniform_(-y, y)
        
    def forward(self, x):
        identity = x
        out = self.block(x)

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        out = self.relu(out)

        return out

class TransformerModel_XYZ(nn.Module):
    def __init__(self, input_dim: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()

        self.input_dim = input_dim
        
        self.model_type = 'Transformer'
        
        self.xyz_encoder = nn.Linear(input_dim, d_model)

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        
        self.d_model = d_model

        # classification decoder
        self.fc1 = nn.Linear(d_model, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.01
        self.xyz_encoder.bias.data.zero_()
        self.xyz_encoder.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc3.bias.data.zero_()
        self.fc3.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, lengths: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, D]
            lengths: Tensor, shape [batch_size, 1]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        # every timestep until the object is never seen again

        # make x the right shape
        x = self.relu(self.xyz_encoder(src))
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x)

        # start classifying the output of the encoder
        print(output[lengths.permute(1,0)].shape)
        quit()
        output = self.relu(self.fc1(output[:, -1]))
        
        output = self.relu(self.fc2(output))
        output = self.sig(self.fc3(output))

        return output
    
class TransformerModel_XYZRGBD_Mask(nn.Module):
    def __init__(self, input_dim: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()

        self.input_dim = input_dim
        
        self.model_type = 'Transformer'
        
        self.concat_layer = nn.Linear(input_dim, d_model)
        
        # in_channels - R,G,B,D,OCCL_MASK,OBJ_MASK, out_channels - 1, 
        self.img_encoder = nn.Sequential(
                # nn.Conv2d(4, 128, (3, 3), 1, padding=(1,1)),
                # nn.ReLU(),
                # Block(128, 128),
              nn.Conv2d(5, 8, (3,3), 1, padding=(1,1)),
              nn.ReLU(),
              nn.Conv2d(8, 64, (3,3), 1),
              nn.ReLU(),
              nn.MaxPool2d((3,3)),
                # nn.AvgPool2d((100,100))
        )

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        
        self.d_model = d_model

        # classification decoder
        self.fc1 = nn.Linear(d_model, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

        self.init_weights()

    def init_cnn_weights(self, m) -> None:
        if isinstance(m, nn.Conv2d):
            initrange = 0.05
            m.weight.data.uniform_(-initrange, initrange)

    def init_weights(self) -> None:
        initrange = 0.05
        self.concat_layer.bias.data.zero_()
        self.concat_layer.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc3.bias.data.zero_()
        self.fc3.weight.data.uniform_(-initrange, initrange)
        self.img_encoder.apply(self.init_cnn_weights)

    def forward(self, src: Tensor, timesteps: Tensor, input_images: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size, D]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = src.squeeze(0)
        # every timestep until the object is never seen again
        inputs = torch.full((src.size(1), self.input_dim - 3), 0.).cuda()
        for j, t in enumerate(timesteps.squeeze(0)):
            # grab the frames
            img_enc = self.img_encoder(input_images[:, j].cuda())

            inputs[t.int() - 1, :] = img_enc.view(-1)

        # make x the right shape
        x = src.unsqueeze(0).permute(1, 0, 2)
        
        x = torch.cat((inputs.unsqueeze(1), x), axis=-1)
        x = self.relu(self.concat_layer(x))

        x = self.pos_encoder(x)
        x_mask = generate_square_subsequent_mask(x.size(1)).cuda()
        output = self.transformer_encoder(x, x_mask)

        # start classifying the output of the encoder
        output = self.relu(self.fc1(output.squeeze(1)[-1, :]))
        output = self.relu(self.fc2(output))
        output = self.sig(self.fc3(output))

        return output

class TransformerModel_XYZRGBD(nn.Module):
    def __init__(self, input_dim: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()

        self.input_dim = input_dim
        
        self.model_type = 'Transformer'
        
        self.xyz_encoder = nn.Linear(input_dim, d_model)
        
        # in_channels - R,G,B,D,OCCL_MASK,OBJ_MASK, out_channels - 1, 
        self.img_encoder = nn.Sequential(
                # nn.Conv2d(4, 128, (3, 3), 1, padding=(1,1)),
                # nn.ReLU(),
                # Block(128, 128),
              nn.Conv2d(4, 8, (3,3), 1, padding=(1,1)),
              nn.ReLU(),
              nn.Conv2d(8, 64, (3,3), 1),
              nn.ReLU(),
              nn.MaxPool2d((3,3)),
                # nn.AvgPool2d((100,100))
        )

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        
        self.d_model = d_model

        # classification decoder
        self.fc1 = nn.Linear(d_model, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

        self.init_weights()

    def init_cnn_weights(self, m) -> None:
        if isinstance(m, nn.Conv2d):
            initrange = 0.1
            m.weight.data.uniform_(-initrange, initrange)

    def init_weights(self) -> None:
        initrange = 0.1
        self.xyz_encoder.bias.data.zero_()
        self.xyz_encoder.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc3.bias.data.zero_()
        self.fc3.weight.data.uniform_(-initrange, initrange)
        self.img_encoder.apply(self.init_cnn_weights)

    def forward(self, src: Tensor, timesteps: Tensor, input_images: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size, D]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        results = []
        src = src.squeeze(0)
        for i in range(src.size(0)):
            # every timestep until the object is never seen again
            inputs = torch.full((src.size(1), self.input_dim - 3), 0.).cuda()
            for j, t in enumerate(timesteps.squeeze(0)):
                # grab the frames
                img_enc = self.img_encoder(input_images[:, j].cuda())

                inputs[t.int() - 1, :] = img_enc.view(-1)

            # make x the right shape
            x = src[i]
            x = x.unsqueeze(0).permute(1, 0, 2)
            
            x = torch.cat((inputs.unsqueeze(1), x), axis=-1)
            x = self.relu(self.xyz_encoder(x))

            x = self.pos_encoder(x)
            x_mask = generate_square_subsequent_mask(x.size(1)).cuda()
            output = self.transformer_encoder(x, x_mask)

            # start classifying the output of the encoder
            output = self.relu(self.fc1(output.squeeze(1)[-1, :]))
            output = self.relu(self.fc2(output))
            output = self.sig(self.fc3(output))
            results.append(output)

        output = torch.stack(results, dim=0)
        return output

def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model%2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:,0:-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)