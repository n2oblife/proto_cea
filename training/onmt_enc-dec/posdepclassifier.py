import torch
import torch.nn as nn
import onmt.decoders as dec

from trankit.utils.base_utils import *
from trankit.utils.conll import *
from trankit.models import Deep_Biaffine
from trankit.tpipeline import TPipeline
from onmt.utils.misc import use_gpu


@dec.register_decoder(name='posdep')
class PosDepDecoder(dec.DecoderBase, TPipeline):
    def __init__(self, opt:dict, embeddings = None, attentional=True) -> None:
        dec.DecoderBase(PosDepDecoder, self).__init__(attentional)

        # trankit's config for TPipeline, parameters are : self._param
        tconfig={
            'category': 'customized', # pipeline category
            'task': opt.task, # task name
            'save_dir': opt.save_data, # directory for saving trained model
            'gpu' : use_gpu(opt),
            'max_epoch': opt.train_steps,
            'train_conllu_fpath': opt.data.corpus.path_tgt, # annotations file in CONLLU format  for training
            'dev_conllu_fpath': opt.data.valid.path_tgt # annotations file in CONLLU format for development
            }
        TPipeline(PosDepDecoder,self).__init__(training_config=tconfig)

        



    @classmethod
    def from_opt(cls, opt, embeddings):
        return cls(opt, embeddings)