import os
import torch
import torch.nn as nn
import onmt.encoders as enc

from transformers import XLMRobertaModel # there are problems with the adapters
from transformers.adapters import AdapterConfig, AdapterType, XLMRobertaAdapterModel
from trankit.utils.base_utils import word_lens_to_idxs_fast
from onmt.utils.misc import use_gpu


@enc.register_encoder(name='xlmr')
class XlmrEncoder(enc.EncoderBase) :
    def __init__(self, opt:dict, embeddings = None, model_name:str = 'tagger') -> None:
        # model_name : embedding ?
        super(XlmrEncoder, self).__init__()
        # BaseModel
        # xlmr encoder
        self.xlmr_dim = 768 if opt.embedding_name == 'xlm-roberta-base' else 1024
        self.xlmr = XLMRobertaModel.from_pretrained(opt.embedding_name, # XLMRobertaAdapterModel ?
                                                    cache_dir=os.path.join(opt._cache_dir, opt.embedding_name), # might cause issues
                                                    output_hidden_states=True)
        self.xlmr_dropout = nn.Dropout(p=opt.embedding_dropout)
        # add task adapters
        task_config = AdapterConfig.load("pfeiffer",
                                         reduction_factor=6 if opt.embedding_name == 'xlm-roberta-base' else 4)
        self.xlmr.add_adapter(opt.task_name, AdapterType.text_task, config=task_config)
        self.xlmr.train_adapter([opt.task_name])
        self.xlmr.set_active_adapters([opt.task_name])
        # embeddings for onmt ?
        self.embeddings = embeddings  # TODO : see if needed for the model type seq2seq

    @classmethod
    def from_opt(cls, opt:dict, embeddings=None):
        """Alternate constructor from the option file"""
        return cls(opt, embeddings)

    def forward(self, src, src_len=None):

        return self.get_tagger_inputs(self, src)

    def update_dropout(self, dropout, attention_dropout):
        self.xlmr_dropout = nn.Dropout(p = dropout)
        for layer in self.xlmr :
            layer.update_dropout(dropout, attention_dropout)

    def encode_words(self, piece_idxs, attention_masks, word_lens):
        batch_size, _ = piece_idxs.size()
        all_xlmr_outputs = self.xlmr(piece_idxs, attention_mask=attention_masks)
        xlmr_outputs = all_xlmr_outputs[0] # [batch_size, word_lens, xlmr_dim]
        cls_reprs = xlmr_outputs[:, 0, :].unsqueeze(1)  # [batch size, 1, xlmr dim]
        # average all pieces for multi-piece words
        idxs, masks, token_num, token_len = word_lens_to_idxs_fast(word_lens)
        idxs = piece_idxs.new(idxs).unsqueeze(-1).expand(batch_size, -1, self.xlmr_dim) + 1
        masks = xlmr_outputs.new(masks).unsqueeze(-1)
        xlmr_outputs = torch.gather(xlmr_outputs, 1, idxs) * masks
        xlmr_outputs = xlmr_outputs.view(batch_size, token_num, token_len, self.xlmr_dim)
        xlmr_outputs = xlmr_outputs.sum(2)
        return xlmr_outputs, cls_reprs
    
    def get_tagger_inputs(self, batch):
        # encoding
        word_reprs, cls_reprs = self.encode_words(
            piece_idxs=batch.piece_idxs,
            attention_masks=batch.attention_masks,
            word_lens=batch.word_lens
        )
        return word_reprs, cls_reprs