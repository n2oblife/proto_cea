import torch
from trankit import Pipeline, TPipeline

path_to_tagging = 'Projet/proto_utils/save_dir/first_trankit_training_posdep/customized.tagger.mdl'
batch_sentences_str = "Hello I'm a single sentence. And another sentence. And the very very last one"

config={
    'category': 'customized', # pipeline category
    'task': 'posdep', # task name
    'save_dir': '../proto_utils/save_dir', # directory for saving trained model
    'gpu' : torch.cuda.is_available(),
    'max_epoch':150,
    'train_conllu_fpath': main_factoryAI_path+'proto_utils/datasets/ud-treebanks-v2.10-trainable/UD_English-EWT/en_ewt-ud-train.conllu', # annotations file in CONLLU format  for training
    'dev_conllu_fpath': main_factoryAI_path+'proto_utils/datasets/ud-treebanks-v2.10-trainable/UD_English-EWT/en_ewt-ud-dev.conllu' # annotations file in CONLLU format for development
    }
model = TPipeline(config) 
weights = torch.load(path_to_tagging, map_location='cpu')

model