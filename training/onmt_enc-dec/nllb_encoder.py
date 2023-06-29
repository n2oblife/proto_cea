"""
Implementation of the NllbEncoder to be used
as interface by onmt trainer
"""
import torch
from transformers import NllbTokenizer
from onmt.encoders.encoder import EncoderBase
import onmt.encoders as enc

@enc.register_encoder(name='nllb')
class NllbEncoder(EncoderBase):
    def __init__(self, embeddings) -> None:
        """
            Args :
        """
        super(NllbEncoder, self).__init__()
        self._tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        self.embeddings = embeddings


    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        breakpoint()
        return cls(embeddings)


    def forward(self, src: torch.tensor, batch_size : int | None):
        breakpoint()
        #return self._tokenizer(src), None, batch_size
        return src, None, batch_size

    def update_dropout(self, dropout, attention_dropout):
        self._tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M",
                                                        dropout = dropout,
                                                        attention_dropout = attention_dropout)