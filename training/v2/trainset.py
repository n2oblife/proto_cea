from trankit import TPipeline
from trankit.tpipeline import *


# trankit's config for TPipeline, parameters are : self._param
tconfig={
    'category': 'customized', # pipeline category
    'task': 'posdep', # task name
    'save_dir': '.', # directory for saving trained model
    'gpu' : False,
    'max_epoch': 2,
    'train_conllu_fpath': '/home/zk274707/Projet/datasets/ud-treebanks-v2.10-trainable/UD_English-EWT/en_ewt-ud-train.conllu', # annotations file in CONLLU format  for training
    'dev_conllu_fpath': '/home/zk274707/Projet/datasets/ud-treebanks-v2.10-trainable/UD_English-EWT/en_ewt-ud-train.conllu' # annotations file in CONLLU format for development
    }

tp = TPipeline(training_config=tconfig)
train_set = TPipeline(training_config=tconfig)._prepare_posdep()
tp._tagger
print(train_set)