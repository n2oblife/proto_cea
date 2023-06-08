import os
import onmt.bin.train as tr
from onmt.utils.parse import ArgumentParser
import yaml
from yaml import SafeLoader, SafeDumper



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