import os
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import NllbTokenizer, XLMRobertaTokenizer
from trankit import Pipeline, TPipeline
import trankit.models.base_models as base_models
import trankit.config as conf


main_loc_path = '/home/zk274707/Projet/'
main_factoryAI_path = '/home/users/zkanit/proto_utils/'

print('gpu available : ',torch.cuda.is_available())

batch_sentences_list = ["Hello I'm a single sentence.", "And another sentence.", "And the very very last one"]
batch_sentences_str = "Hello I'm a single sentence. And another sentence. And the very very last one"

configuration = conf.Config()
configuration._cache_dir = '../proto_utils/cache/trankit/'

pipe = Pipeline('auto', cache_dir='../proto_utils/cache/')
posdep = pipe.posdep(batch_sentences_list, is_sent=True)
print('pos dep of Trankit :',posdep)

# revoir les chemins une fois sur factoryAI

path_to_toks = '../proto_utils/save_dir/HG/'
nlb_tokenizer = torch.load(path_to_toks+'nllb_tok.pt')
xlm_tokenizer = torch.load(path_to_toks+'xml_roberta_tok.pt')


# emb = base_models.Multilingual_Embedding(config=configuration)
# output = emb.get_tagger_inputs(batch_sentences_str)
# print(output)