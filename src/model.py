import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from dataclasses import dataclass

@dataclass
class Tacotron2Config:

    ### Mel input features ####
    num_mels = 80

    ### Character Embeddings ###
    character_embed_dim: int = 512
    num_chars :int = 67
    pad_token_id: int = 0

    ## Encoder config ###
    encoder_kernel_size: int = 5
    enocoder_n_convulations: int = 3
    encoder_embed_dim: int = 52
    encoder_dropout_p: float =0.5

    ### Decoder config ###
    decoder_embed_dim: int = 1024
    decoder_prenet_dim: int = 256
    decoder_prenet_depth: int = 2
    decoder_prenet_dropout_p: float = 0.5
    decoder_postnet_num_convs: int = 5
    decoder_postnet_n_filters: int = 512
    decoder_postnet_kernel_size: int = 5
    decoder_postnet_dropout_p: float = 0.5
    decoder_dropout_p: float = 0.1

    ### Attention confign ###

    attention_dim : int = 128
    attention_location_n_filters: int = 32
    attention_location_kernel_size: int = 31
    attention_dropout_p: float = 0.1

class LinearNorm(nn.Module):
    """Standard Linear Layer with different initialization Stratergies"""

    def __init__(self,
                 in_features,
                 out_features,
                 bias = True,
                 w_init_grain = "linear"):

        super(LinearNorm, self).__init__()

        self.linear = nn.Linear( in_features, out_features, bias = bias)

        torch.nn.init.xavier_uniform_(
            self.linear.weight,
            gain = torch.nn.init.calculate_gain(w_init_grain)
        )

    def forward (self, x):
        return self.linear(x)

class ConvNorm(nn.Module):

    """"Standard Convulation layer with different Initialzation strateriges"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size = 1,
                 stride = 1,
                 padding = None,
                 dilation = 1,
                 bias = True,
                 w_init_grain = "linear"):

        super(ConvNorm, self).__init__()

        if padding is None:
            padding = "same"

        self.conv = nn.Convld( in_channels, out_channels, kernel_size= kernel_size,
                              stride=stride, padding= padding, dilation = dilation,
                              bias = bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight,
            gain = torch.nn.init.calculate_gain(w_init_grain)
        )

    def forward (self, x):
        return self.conv(x)

