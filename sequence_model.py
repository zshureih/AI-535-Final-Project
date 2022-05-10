import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from torch.utils.data import dataset

class TransformerModel_XYZRGBD(nn.Module):

    def __init__(self, input_dim: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()

        self.input_dim = input_dim
        
        self.model_type = 'Transformer'
        
        self.xyz_encoder = nn.Linear(input_dim, d_model)
        
        # in_channels - R,G,B,D,OCCL_MASK,OBJ_MASK, out_channels - 1, 
        self.img_encoder = nn.Sequential(
              nn.Conv2d(4, 8, (3,3), 1, padding=(1,1)),
              nn.ReLU(),
              nn.Conv2d(8, 64, (3,3), 1),
              nn.ReLU(),
              nn.MaxPool2d((3,3)),
        )

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        
        self.d_model = d_model

        # classification decoder
        self.decoder = nn.Linear(self.d_model, 3)
        self.relu = nn.ReLU()

        self.init_weights()

    def init_cnn_weights(self, m) -> None:
        if isinstance(m, nn.Conv2d):
            initrange = 0.1
            m.weight.data.uniform_(-initrange, initrange)

    def init_weights(self) -> None:
        initrange = 0.1

        self.xyz_encoder.bias.data.zero_()
        self.xyz_encoder.weight.data.uniform_(-initrange, initrange)

        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

        self.img_encoder.apply(self.init_cnn_weights)


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
            time = timesteps[i, :lengths[i, 1].long()]
            x = src[i, :].unsqueeze(0).cuda()
            images = input_images[i, :lengths[i, 1].long()]

            # start processing our images
            inputs = torch.full((x.size(1), self.input_dim - 3), 0.).cuda()
            img_enc = self.img_encoder(images.cuda())
            flattened_enc = torch.reshape(img_enc, (img_enc.shape[0], self.input_dim - 3))
            inputs[time.long().squeeze() - 1, :] = flattened_enc

            # concat image encodings and xyz values, |S| x 1 x input_dim
            x = torch.cat([inputs.unsqueeze(0), x], axis=-1).permute(1, 0, 2)
            
            # linear + positional encoding, |S| x 1 x d_model
            x = self.relu(self.xyz_encoder(x))
            x = self.pos_encoder(x)

            # This is super important according to experiments
            x_mask = generate_square_subsequent_mask(x.size(0)).cuda()
            output = self.transformer_encoder(x, x_mask)
            
            # transform the output of the encoder back into xyz, |S| x 1 x 3
            output = self.decoder(output)

            results.append(output)

        results = torch.stack(results, dim=0)

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