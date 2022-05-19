from email.mime import image
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from torch.utils.data import dataset
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np

MASK = 99.
PAD = -99.

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

class TransformerModel_XYZRGBD(nn.Module):

    def __init__(self, input_dim: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()

        self.input_dim = input_dim
        
        self.model_type = 'Transformer'
        
        self.concat_encoder = nn.Linear(input_dim - 3 + d_model, d_model)
        self.xyz_encoder = nn.Linear(3, d_model)

        # use pretrained resnet18
        self.img_encoder = models.resnet18(pretrained=True)

        # freeze everything except for the input
        for param in self.img_encoder.parameters():
            param.requires_grad = False
        self.img_encoder.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2,2), padding=(3,3), bias=False).requires_grad_(True)

        # remove the fully connected layer at the end
        self.img_encoder = torch.nn.Sequential(*(list(self.img_encoder.children())[:-1]))

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        
        self.d_model = d_model

        # classification decoder
        self.xyz_decoder = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )
        self.relu = nn.ReLU()

        self.init_weights()

    def init_cnn_weights(self, m) -> None:
        if isinstance(m, nn.Conv2d):
            initrange = 0.1
            m.weight.data.uniform_(-initrange, initrange)
    
    def init_linear_weights(self, m) -> None:
        if isinstance(m, nn.Linear):
            initrange = 0.1
            m.bias.data.zero_()
            m.weight.data.uniform_(-initrange, initrange)

    def init_weights(self) -> None:
        initrange = 0.1

        self.xyz_encoder.bias.data.zero_()
        self.xyz_encoder.weight.data.uniform_(-initrange, initrange)

        self.decoder.apply(self.init_linear_weights)


    def forward(self, src: Tensor, timesteps: Tensor, input_images: Tensor, lengths: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size, 3]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, 3]
        """

        # src = self.encoder(src)
        # src = self.pos_encoder(src)
        # e_output = self.transformer_encoder(src, src_mask)
        # output = self.decoder(e_output)

        # return output

        results = []
        # for each scene in batch:
        for i in range(src.size(0)):

            # make x the right shape by trimming the extra padding beyond scene length
            x = src[i, :].unsqueeze(0).cuda()
            unpadded_idx = torch.where(x != PAD)
            unpadded_idx = torch.unique(unpadded_idx[1])
            unpadded_x = x[:, unpadded_idx, :]
            
            unmmask_idx = torch.where(unpadded_x != MASK)
            unmmask_idx = torch.unique(unmmask_idx[1])

            images = input_images[i, :lengths[i, 1].long()]

            # start processing our images
            inputs = torch.full((unpadded_x.size(1), self.input_dim - 3), 0.).cuda()
            img_enc = self.img_encoder(images.cuda())
            flattened_enc = torch.reshape(img_enc, (img_enc.shape[0], self.input_dim - 3))
            inputs[unmmask_idx, :] = flattened_enc

            x = self.relu(self.xyz_encoder(unpadded_x))

            # concat image encodings and xyz values, |S| x 1 x input_dim
            x = torch.cat([inputs.unsqueeze(0), x], axis=-1).permute(1, 0, 2)

            # linear + positional encoding, |S| x 1 x d_model
            x = self.relu(self.concat_encoder(x))
            x = self.pos_encoder(x)

            # This is super important according to experiments
            x_mask = generate_square_subsequent_mask(x.size(0)).cuda()
            output = self.transformer_encoder(x, x_mask)
            
            # transform the output of the encoder back into xyz, |S| x 1 x 3
            output = self.xyz_decoder(output)

            results.append(output)

        return results

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