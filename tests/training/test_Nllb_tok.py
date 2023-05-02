import sys
sys.path.insert(1, '/home/zk274707/Projet/proto/')
import os
import torch
from transformers import NllbTokenizer
from trankit import Pipeline, TPipeline
import trankit.models.base_models as base_models
import trankit.config as conf
from utils.python.utils import *

main_loc_path = '/home/zk274707/Projet/'
main_factoryAI_path = '/home/users/zkanit/proto_utils/'

print('gpu available : ',torch.cuda.is_available())

#tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
tok_path_factoryAI = '../proto_utils/save_dir/HG/'
tokenizer = torch.load(tok_path_factoryAI+'nllb_tok.pt')

print(tokenizer.model_max_length)

#my_var = NllbTokenizer.max_len_single_sentence


batch_sentences_list = ["Hello I'm a single sentence.", "And another sentence.", "And the very very last one"]
batch_sentences_str = "Hello I'm a single sentence. And another sentence. And the very very last one"
# tokenized = tokenizer()
docu = tokenizer(batch_sentences_list, padding = 'max_length')

# docu.wordl_lens = adapt_nllb_to_trankit(docu)

print(docu.keys())

# configuration = conf.Config()
# configuration._cache_dir = '../proto_utils/cache/trankit/tpPipeline'
# bse_mdl = base_models.Base_Model(config=configuration, task_name='posdep')

# pipe = Pipeline('auto') # Rq : 'gpu = False / dans le cas du taff sur cpu meme si ca marche pas

input_hidden_arc_dim = 128
UD_n_labels = 37 
DB = base_models.Deep_Biaffine(tokenizer.model_max_length, # input dimmension is 1024 here
                       tokenizer.model_max_length,
                       input_hidden_arc_dim,
                       UD_n_labels)
in_ids = torch.tensor(docu['input_ids'])
print(in_ids.dtype)
att_mask = torch.tensor(docu['attention_mask'])
print(att_mask.dtype)
print(DB(in_ids, att_mask))

# = torch.tensor(docu) # need to use the get tagger from the base model and ecode_words function


# training_config={
#     #'category': 'customized-mwt-ner', # pipeline category
#     'task': 'posdep', # task name
#     'save_dir': '../proto_utils/save_dir', # directory for saving trained model
#     #'gpu' : False,
#     'train_conllu_fpath': main_factoryAI_path+'datasets/ud-treebanks-v2.10-trainable/UD_English-EWT/en_ewt-ud-train.conllu', # annotations file in CONLLU format  for training
#     'dev_conllu_fpath': main_factoryAI_path+'datasets/ud-treebanks-v2.10-trainable/UD_English-EWT/en_ewt-ud-dev.conllu' # annotations file in CONLLU format for development
#     }

# # initialize a trainer for the task
# trainer = TPipeline(training_config)

# trainer.train()

#out = trainer(batch_sentences_str)

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