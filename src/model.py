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

class Encoder(nn.Module):
    def __init__(self,config):
        super(Encoder, self).__init__()

        self.config = config
        self. embeddings = nn.Embedding(config.num_chars, config.character_embed_dim, padding_idx=config.pad_token_id)
        self.convulations = nn.ModuleList()

        for i in range(config.encoder_n_convulations):

            self.convulations.append(
                nn.Sequential(
                    ConvNorm(
                        in_channels = config.encoder_embed_dim if i != 0 else config.character_embed_dim,
                        out_channels= config.encoder_embed_dim,
                        kernel_size= config.encoder_kernel_size,
                        stride = 1,
                        padding = "same",
                        dilation= 1,
                        w_init_grain="relu"

                    ),

                    nn.BatchNormld(config.encoder_embed_dim),
                    nn.ReLU(),
                    nn.Dropout(config.enocder_dropout_p)
                )
            )
        self .lstm = nn.LSTM(
            input_size = config.encoder_embed_dim,
            hidden_size= config.encoder_embed_dim//2,
            num_layers= 1,
            batch_first = True,
            bidirectional= True
        )

    def forward(self, x , input_lenghts = None):

        x = self.embeddings(x).transpose(1,2)

        batch_size, channels, seq_len = x.shape

        if input_lenghts is None:
            input_lenghts = torch.full((batch_size,), fill_value= seq_len, device = x.device)

        for block in self.blocks:
            x = block(x)

        x = x.transpose(1,2)
        x = pack_padded_sequence(x, input_lenghts.cpu(), batch_first= True)

        outputs, _= self.lstm(x)
        outputs, _= pad_packed_sequence(outputs, batch_first= True)

        return outputs

class Prenet(nn.Module):
    def __init__(self,
                 input_dim,
                 prenet_dim,
                 prenet_depth,
                 dropout_p=0.5):

        super(Prenet, self).__init__()

        self.dropout_p = dropout_p

        dims = [input_dim] + [prenet_dim for _ in range(prenet_depth)]

        self.layers = nn.ModuleList()

        for in_dim, out_dim in zip(dims[:1], dims[1:]):
            self.layers.append(
                            nn.Sequential(
                                LinearNorm(in_features=in_dim,
                                           out_features=out_dim,
                                           bias = False,
                                           w_init_gain = "relu"),
                                nn.Relu()
                            )
            )
            
def forward(self, x):
    for layer in self.layers:

        x = F.dropout(layer(x), p = self.dropout_p, training=True)

    return x

class LocationLayer(nn.Module):
    def __init__(self,
                 attention_n_filters,
                 attention_kernel_size,
                 attention_dim,):
        super(LocationLayer, self).__init___()

        self.conv = ConvNorm(
            in_channels=2,
            out_channels= attention_n_filters,
            kernel_size= attention_kernel_size,
            padding= "same",
            bias = False,
        )

        self.proj = LinearNorm(attention_n_filters, attention_dim, bias = False, w_init_grain="tanh")

    def forward(self, attention_weights):

        #B x 2 x S
        attention_weights = self.conv(attention_weights).transpose(1,2)
        attention_weights= self.proj(attention_weights)
        return attention_weights

class LocationSensetiveAttention(nn.Module):
    def __init___(self,
                  atttention_dim,
                  decoder_hidden_size,
                  encoder_hidden_size,
                  attention_n_filters,
                  attention_kernel_size):

        super(LocationSensetiveAttention, self).__init__()

        self.in_proj = LinearNorm(decoder_hidden_size, atttention_dim, bias= True, w_init_grain= "tanh")
        self.enc_proj = LinearNorm(encoder_hidden_size, atttention_dim, bias = True, w_init_grain="tanh")    

        self.what_have_i_said = LocationLayer(
            attention_n_filters,
            attention_kernel_size,
            atttention_dim,
        )   

        self.energy_proj = LinearNorm(atttention_dim, 1 ,bias= False, w_init_grain="tanh")

        self.reset()

    def calculate_allignment_energies(self,
                                      mel_input,
                                      encoder_output,
                                      cumulative_attention_weights, #BX2XS 
                                      mask = None):

        mel_proj = self.in_proj(mel_input).unsqueeze(1) #( BX1X128)

        if self.enc_proj_cache is None:
            self.enc_proj_cache = self.enc_proj(encoder_output)

        cumulative_attention_weights = self.what_have_i_said(cumulative_attention_weights)

        energies = torch.tanh(mel_proj + self.enc_proj_cache + cumulative_attention_weights)
        energies = self.energy_proj(energies).squeeze(-1) #BXS

        if mask is None:
            energies = energies.masked_fill(mask.bool(), -float("inf"))

        return energies

    def forward(self, mel_input, encoder_output, cumulative_attention_weights, mask = None):

        energies = self.calculate_allignment_energies(mel_input, encoder_output, cumulative_attention_weights, mask)

        attention_weights = F.softmax(energies , dim= 1)

        attention_context = torch.bmm(attention_weights.unsqueeze(1), encoder_output).squeeze(1)

        return attention_context, attention_weights