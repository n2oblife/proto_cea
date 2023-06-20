import numpy as np
import matplotlib.pyplot as plt

import sys

import torch 
import torch.nn as nn
from collections import OrderedDict

from torch.utils.data import DataLoader
import torch.optim as optim

def get_var_from_memory(var : any) -> any:
    '''Not always working'''
    # Get the memory address of the variable
    var_address = id(var)

    # Get the namespace where the variable is defined
    for ns in globals(), locals():
        if var_address in ns.values():
            namespace = ns
            break

    # Access the variable using its name in the namespace
    return namespace["var"]

def adapt_nllb_to_trankit(batch:list) -> list:
    '''Adapt the batch from the output of the Nllb tokenizer to be usable by the trankit dep parser'''
    # complete batch avec le word lens
    pieces = [[p for p in self.wordpiece_splitter.tokenize(w) if p != '▁'] for w in words]
    for ps in pieces:
        if len(ps) == 0:
            ps += ['-']
    word_lens = [len(x) for x in pieces]
    return word_lens

def load_seed():
    print('Setting up seed...')
    seed  = 2147483647
    # set random seed
    try :
        os.environ['PYTHONHASHSEED'] = str(seed)
    except :
        pass
    try :
        random.seed(seed)
    except :
        pass
    try :
        np.random.seed(seed)
    except :
        pass
    try :
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # empty cache
        torch.cuda.empty_cache()
    except :
        pass

class Device():
    def __init__(self) -> None:
        import torch

        if torch.cuda.is_available():
            # FactotyAI
            base_dir = '/home/users/zkanit/'
        else :
            # PC
            base_dir = '/home/zk274707/Projet/proto/'
        
        self._base_dir = base_dir
        self._save_dir = base_dir+''
        self._data = base_dir+''
        self._gpu = torch.cuda.is_available()training/v1/onmt_tuto_config.yamltraining/v1/onmt_tuto_config.yaml
        