import os
import onmt.bin.train as tr
from onmt.utils.parse import ArgumentParser
import yaml
from yaml import SafeLoader, SafeDumper

# save all parameters from the yaml file

class Config():
    def __init__(self) -> None:
        self.input_size = 1024
        self.word_vec_size = 600 #to change according to 1.6*sqrt(unique_elts)
        self.word_vocab_size = 600
        self.word_padding_idx = 1
        self.embeddings = 
        self.max_relative_positions = 0
        self.dropout = 0.3
        self.attn_dropout = 0.1
        self.lr = 0.001
        self.mometum = 0.9
        self.num_epoch = 20
        self.batch_size = 32
        self.copy_attn = False
        self.self_attn_type = 'scaled_dot'
        self.aan_useffn = True
        self.full_ctxt_alignt = True
        self.aan_useffn = False
        self.full_context_alignment = True
        self.alignment_layer = 1
        self.alignment_heads = 1  

def load_default_opt() -> dict :
    parser = tr._get_parser()
    opt, unknown = parser.parse_known_args()
    ArgumentParser.update_model_opts(opt)
    ArgumentParser.validate_model_opts(opt)

    opt.dump_samples = False
    return opt

def update_yaml(yaml_path : str, key :str, changes : str, share_vocab : bool = False) -> None :
    return NotImplemented

def build_onmt_config(yaml_path : str, lgge:str, device:Device_CEA) -> dict:
    config = 
    return NotImplemented

def build_vocab(yaml_path:str, config:dict) -> None:
    # write in the bash script the correct files
    # correct the corpus opts
    print('bash onmt_training')
    os.system('bash onmt_training.sh') 
    return NotImplemented

def dump_in_yaml(yaml_path : str, config : dict) -> None :
    assert yaml_path[-5:] == '.yaml', "configuration file must be .yaml"
    with open(yaml_path, 'w') as f:
        data = yaml.dump(config, f, sort_keys=False, default_flow_style=False)