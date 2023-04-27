import os
import torch
from transformers import NllbTokenizer
from trankit import Pipeline, TPipeline
import trankit.models.base_models as base_models
import trankit.config as conf


tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

#print(tokenizer.model_max_length)

#my_var = NllbTokenizer.max_len_single_sentence



batch_sentences_list = ["Hello I'm a single sentence.", "And another sentence.", "And the very very last one"]
batch_sentences_str = "Hello I'm a single sentence. And another sentence. And the very very last one"
docu = tokenizer(batch_sentences_list, padding = 'max_length')

print(docu.keys())

configuration = conf.Config()
configuration._cache_dir = './cache/trankit/tpPipeline'
bse_mdl = base_models.Base_Model(config=configuration, task_name='posdep')

pipe = Pipeline('auto',gpu=False)

input_hidden_arc_dim = 128
UD_n_labels = 37 
DB = base_models.Deep_Biaffine(tokenizer.model_max_length, # input dimmension is 1024 here
                       tokenizer.model_max_length,
                       input_hidden_arc_dim,
                       UD_n_labels)

# = torch.tensor(docu) # need to use the get tagger from the base model and ecode_words function
print()

training_config={
    #'category': 'customized-mwt-ner', # pipeline category
    'task': 'posdep', # task name
    'save_dir': './save_dir', # directory for saving trained model
    'train_conllu_fpath': '/home/zk274707/Projet/datasets/ud-treebanks-v2.10-trainable/UD_English-EWT/en_ewt-ud-train.conllu', # annotations file in CONLLU format  for training
    'dev_conllu_fpath': '/home/zk274707/Projet/datasets/ud-treebanks-v2.10-trainable/UD_English-EWT/en_ewt-ud-dev.conllu', # annotations file in CONLLU format for development
    'gpu' : False,
    }

# initialize a trainer for the task
trainer = TPipeline(training_config)

trainer.train()

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