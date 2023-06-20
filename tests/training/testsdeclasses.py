import inspect


import torch
from trankit.tpipeline import TPipeline
import onmt.decoders as dec

config={
    'category': 'customized', # pipeline category
    'task': 'posdep', # task name
    'save_dir': '../proto_utils/save_dir', # directory for saving trained model
    'gpu' : torch.cuda.is_available(),
    'max_epoch':150,
    'train_conllu_fpath': '/home/zk274707/Projet/datasets/ud-treebanks-v2.10-trainable/UD_English-EWT/en_ewt-ud-train.conllu',
    'dev_conllu_fpath': '/home/zk274707/Projet/datasets/ud-treebanks-v2.10-trainable/UD_English-EWT/en_ewt-ud-dev.conllu'
    }
model = TPipeline(config) 

deco = dec.DecoderBase()

'''
attributes = inspect.getmembers(MyClass, lambda a:not(inspect.isroutine(a)))
print([a for a in attributes if not(a[0].startswith('__') and a[0].endswith('__'))])
'''


print("TPipelinne : ")
print(model.__dict__.keys())
print('-----------------------')
print("DecoderBase : ")
print(deco.__dict__.keys())

['_lang', '_task', '_train_txt_fpath', '_train_conllu_fpath', '_dev_txt_fpath', '_dev_conllu_fpath', '_train_bio_fpath', '_dev_bio_fpath', '_text_split_by_space', '_save_dir', '_cache_dir', '_gpu', '_use_gpu', '_ud_eval', '_config', 'logger', 'train_set', 'batch_num', 'dev_set', 'dev_batch_num', '_embedding_layers', '_tagger', 'model_parameters', 'optimizer', 'schedule']


['training', '_parameters', '_buffers', '_non_persistent_buffers_set', '_backward_hooks', '_is_full_backward_hook', '_forward_hooks', '_forward_pre_hooks', '_state_dict_hooks', '_load_state_dict_pre_hooks', '_load_state_dict_post_hooks', '_modules', 'attentional']