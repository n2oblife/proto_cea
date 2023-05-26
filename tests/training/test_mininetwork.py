''' File to test the translation of weights from python to C++ '''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from transformers.modeling_utils import PreTrainedModel

class MiniNetwork(nn.Module):
    def __init__(self, in_input_planes : int, in_output_planes : int, in_inter_planes : int) -> nn.Module:
        super(MiniNetwork, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_input_planes, out_channels=in_inter_planes, kernel_size=2)
        self.lin1 = nn.Linear(in_features=in_inter_planes, out_features=in_inter_planes)
        self.conv2 = nn.Conv1d(in_channels=in_inter_planes, out_channels=in_inter_planes, kernel_size=2)
        self.lin2 = nn.Linear(in_features=in_inter_planes, out_features=in_output_planes)

    def forward(self, in_x : torch.tensor) -> torch.tensor:
        out = self.conv1(in_x)
        out = self.lin1(out)
        out = self.conv2(out)
        out = self.lin2(out)
        return out
    
def init_weights(layer) -> None:
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0.1)  

def generate_model() -> nn.Module:
    in_channels, out_channels, inter_planes = 5, 1, 3
    network = MiniNetwork(in_channels, out_channels, inter_planes)
    network.apply(init_weights)
    return network

class Option():
    def __init__(self) -> None:
        self.encoder_type = 'transformer' # transformer or transformer-lm
        self.decoder_type = 'transformer' # transformer or transformer-lm
        self.self_attn_type = 'scaled-dot'
        self.pos_ffn_activation_fn = 'relu' # gelu or fast_gelu or relu
        self.position_encoding = True
        self.max_relative_positions = 0
        self.feat_merge = 'concat' # concat or sum

if __name__ == '__main__' :
    net = generate_model()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    dico = net.state_dict()

    # regularize for opennmt converter
    dico["vocab"] = {"src" : [], "tgt" : []}
    dico["opt"] = Option()

    path = '../proto_utils/save_dir/test_jit/net.pt'
    torch.save(dico, path)
    print(f"dico :  {dico}")
    print('Saved !')


# equivalence avec le fichier de trankit est dico['adapters']
