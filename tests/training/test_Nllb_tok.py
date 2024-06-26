import sys
main_loc_path = '/home/zk274707/Projet/'
main_factoryAI_path = '/home/users/zkanit/'
import torch

print('gpu available : ',torch.cuda.is_available())
if torch.cuda.is_available() :
    sys.path.insert(1, main_factoryAI_path+'proto/')

import os
import torch.nn as nn
from transformers import NllbTokenizer, XLMRobertaTokenizer, XLMRobertaModel
from transformers.adapters import XLMRobertaAdapterModel
from trankit import Pipeline, TPipeline
import trankit.models.base_models as base_models
import trankit.config as conf
# from utils.python.utils import *


# tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
xlm = XLMRobertaAdapterModel.from_pretrained("xlm-roberta-base")
dropout = nn.Dropout(p=0.8)

batch_sentences_list = ["Hello I'm a single sentence.", "And another sentence.", "And the very very last one"]
batch_sentences_str = "Hello I'm a single sentence. And another sentence. And the very very last one"

encoded = tokenizer(batch_sentences_list, 
                    return_tensors='pt',
                    add_special_tokens=True,
                    truncation=True,
                    padding=True) # possibilité de mettre le padding au max
embedded = xlm(encoded['input_ids'], attention_mask = encoded['attention_mask'])
print(embedded[0].size())
print(embedded[0][:, 0, :].unsqueeze(1).size())

# tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
# tok_path_factoryAI = '../proto_utils/save_dir/HG/'
# tokenizer = torch.load(tok_path_factoryAI+'proto_utils/nllb_tok.pt')

# print(tokenizer.model_max_length)

#my_var = NllbTokenizer.max_len_single_sentence


# # tokenized = tokenizer()
# docu = tokenizer(batch_sentences_list, padding = 'max_length')
# output = tokenizer.encode(batch_sentences_str,                
#                         #   add_special_tokens=True,
#                         #   max_length=204,
#                         #   truncation=True
#                           )
# tokenized = tokenizer.tokenize('bonjour, comment va ?')
# print(output)
# print(docu)
# print(output)

# docu.wordl_lens = adapt_nllb_to_trankit(docu)

# print(docu.keys())

# configuration = conf.Config()
# configuration._cache_dir = '../proto_utils/cache/trankit/tpPipeline'
# bse_mdl = base_models.Base_Model(config=configuration, task_name='posdep')

# pipe = Pipeline('auto') # Rq : 'gpu = False / dans le cas du taff sur cpu meme si ca marche pas

# input_hidden_arc_dim = 128
# UD_n_labels = 37 
# DB = base_models.Deep_Biaffine(tokenizer.model_max_length, # input dimmension is 1024 here
#                        tokenizer.model_max_length,
#                        input_hidden_arc_dim,
#                        UD_n_labels)
# in_ids = torch.tensor(docu['input_ids']).to(torch.float) #il faut convertir car int mais les types de DB sont des floats
# print(in_ids.shape)
# att_mask = torch.tensor(docu['attention_mask']).to(torch.float)
# print(att_mask.shape)
# print(DB(in_ids, att_mask)) # pb de taille des matrices au moment du calcul de g1_w pour Deep Biafffine

# = torch.tensor(docu) # need to use the get tagger from the base model and ecode_words function

# Beginning of training ------------------------

# training_config={
#     'category': 'customized', # pipeline category
#     'task': 'posdep', # task name
#     'save_dir': '../proto_utils/save_dir', # directory for saving trained model
#     'gpu' : torch.cuda.is_available(),
#     'max_epoch':150,
#     'train_conllu_fpath': main_loc_path+'datasets/ud-treebanks-v2.10-trainable/UD_English-EWT/en_ewt-ud-train.conllu', # annotations file in CONLLU format  for training
#     'dev_conllu_fpath': main_loc_path+'datasets/ud-treebanks-v2.10-trainable/UD_English-EWT/en_ewt-ud-dev.conllu' # annotations file in CONLLU format for development
#     }

# # # initialize a trainer for the task
# trainer = TPipeline(training_config)

# trainer.train()

# print("---------------- Training DONE ---------------------------")

# treebank = "auto"
# pipe = Pipeline(lang=treebank)
# print(pipe)
# pipe._setup_config(lang=treebank)
# conf = pipe._config
# print(conf)
# classifier = Pipeline.posdep(config=conf, treebank_name=treebank)
# print(classifier)



# pos_dep = trk.pipeline.PosDepClassifier(config=conf, treebank_name='english')
# out_posdep = pos_dep.deprel(batch_sentences)
# print('Sortie du PosDepClassifier :')
# print(out_posdep)

# pipe = trk.Pipeline('auto',gpu=False)
# out_pie = pipe.posdep(batch_sentences)
# print('Sortie de la pipeline :')
# print(type(out_pie))

# dm = trk.Deep_Biaffine()