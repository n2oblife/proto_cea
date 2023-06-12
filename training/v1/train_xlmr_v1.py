from xlmr_model_v1 import *

import torch
import torch.nn as nn
import torch.optim as optim

import onmt
import onmt.models.model as model
import onmt.modules as modules
import onmt.encoders as encoder
import onmt.decoders as decoder
import onmt.trainer as trainer
import onmt.utils.loss as loss
from onmt.utils.scoring_utils import ScoringPreparator
import onmt.opts as opts
import onmt.trainer as trr
import onmt.bin.train as tr
import onmt.train_single as sgl
from onmt.utils.parse import ArgumentParser
import onmt.bin.build_vocab as bld_voc


# -------------------------------------------------------
# Functions

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


# Variables
save_dir = '/home/users/zkanit/proto_utils/save_dir/xlmr_models/'

data_dir = '/home/users/zkanit/proto_utils/datasets/ud-treebanks-v2.10-trainable/UD_English-EWT/'
dat_file_base = 'en_ewt-ud-'
train = '.txt'
test = '.conllu'

# -------------------------------------------------------
# Script

load_seed()

model = Tok_xlmR()
