import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

# Model definition

class DeepBiaffineDecoder(nn.Module):
    def __init__(self, input_dim:int, input_hidden_arc_dim:int, input_includes_root:bool = False) -> nn.Module:
        '''See more on : https://nlp.stanford.edu/pubs/dozat2017deep.pdf
        This class is a Dozat and Manning layer'''
    
        super(DeepBiaffineDecoder, self).__init__()
        # parameters of the model
        self.includes_root = input_includes_root
        self.hidden_arc_dim = input_hidden_arc_dim
        # design of the model
        self.mlp_head = nn.Linear(input_dim,input_hidden_arc_dim)
        self.mlp_dep = nn.Linear(input_dim, input_hidden_arc_dim)
        self.U1 =  torch.randn(input_hidden_arc_dim, input_hidden_arc_dim)
        self.u2 = torch.randn(input_hidden_arc_dim,1)
        self.root = torch.randn(1,1,input_hidden_arc_dim)
        self.root2 = torch.randn(1,1,input_hidden_arc_dim)
    
    def forward(self, input: torch.tensor):
        bacth_size = input.size(1)
        input_t = input.transpose(0,1)

        if (self.includes_root):
            arc_dep = nn.ELU(self.mlp_head(input))
        else :
            
        return output