import torch
from trankit.tpipeline import PosDepClassifier, TPipeline
from trankit.pipeline import Pipeline
import copy
from trankit import xlm


path_to_tagging = 'Projet/proto_utils/save_dir/first_trankit_training_posdep/customized.tagger.mdl'
batch_sentences_str = "Hello I'm a single sentence. And another sentence. And the very very last one"

config={
    'category': 'customized', # pipeline category
    'task': 'posdep', # task name
    'save_dir': '../proto_utils/save_dir', # directory for saving trained model
    'gpu' : torch.cuda.is_available(),
    'max_epoch':150,
    'treebank_name':'english',
    'train_conllu_fpath': '/home/zk274707/Projet/datasets/ud-treebanks-v2.10-trainable/UD_English-EWT/en_ewt-ud-train.conllu', # annotations file in CONLLU format  for training
    'dev_conllu_fpath': '/home/zk274707/Projet/datasets/ud-treebanks-v2.10-trainable/UD_English-EWT/en_ewt-ud-dev.conllu' # annotations file in CONLLU format for development
    }


# tmodel = TPipeline(config)
# model = Pipeline()

# classifier = copy.deepcopy(model._tagger)

# tagger = copy.deepcopy(TPipeline(config))
# weights = torch.load(path_to_tagging, map_location='cpu')

# from onmt.encoders.xlmr_encoder import XLMREncoder
# from onmt.decoders.posdepclassifier import PosDepDecoder

# class model:
#     def __init__(self) -> None:
#         model.encoder = XLMREncoder()
#         model.decoder = PosDepDecoder()

# from trankit.iterators.tagger_iterators import TaggerDataset
# from onmt.utils.misc import TConfig
# import yaml
# from yaml import SafeLoader

# opt = yaml.load('training/v2/local_config.yaml', Loader=SafeLoader)

# data = TaggerDataset(
#     TConfig(opt), config['dev_conllu_fpath'], config['dev_conllu_fpath'], evaluate=False
# )

# print(data[0])

breakpoint()