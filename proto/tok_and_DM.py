import sys
sys.path.append('/home/zk274707/Projet/protos/utils/python')

import torch
import torch.nn as nn
from transformers import NllbTokenizer
from Dozat_Manning import DeepBiaffineDecoder
import trankit.models.base_models as trk

class tok_and_DM(nn.Module):

    def __init__(self, input_hidden_arc_dim:int, input_out_dim: int) -> nn.Module:
        '''Create the 1 prototype according to Victor with a tokenizer and 
        a Dozat and Manning layer
        input_hidden_arc_dim : dimmension of the hidden arc in the Deep Biaffine layer
        input_out_dim : number of  dependency labels
        input'''
        
        super(tok_and_DM, self).__init__()
        self.tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        self.requires_grad = False
        self.DM = trk.Deep_Biaffine(self.tokenizer.model_max_length, # input dimmension is 1024 here
                                    self.tokenizer.model_max_length,
                                    input_hidden_arc_dim,
                                    input_out_dim)
        self.DM.requires_grad = True

    def forward(self, input: str):
        output = self.tokenizer(input, padding= 'max_length')
        output = self.DM(output["input_ids"])
        return output
    

