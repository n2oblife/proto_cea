#!/usr/bin/env python
import sys
sys.path.append('/home/zk274707/Projet/proto/')
from utils.python.utils import load_seed

import torch
import torch.nn as nn
import torch.optim as optim

import onmt
import onmt.models.model as model
import onmt.modules as modules
import onmt.encoders as encoder
import onmt.decoders as decoder
import onmt.trainer as trainer
import onmt.utils.loss as loss
from onmt.utils.scoring_utils import ScoringPreparator
import onmt.opts as opts
import onmt.trainer as trr
import onmt.bin.train as tr
import onmt.train_single as sgl
from onmt.utils.parse import ArgumentParser
import onmt.bin.build_vocab as bld_voc

load_seed()

class Config():
    def __init__(self) -> None:
        self.input_size = 1024
        self.heads = 4
        self.d_ff = 2
        self.num_layers = 3
        self.word_vec_size = 600 #to change according to 1.6*sqrt(unique_elts)
        self.word_vocab_size = 600
        self.word_padding_idx = 1
        self.embeddings = modules.Embeddings(
            word_vec_size=self.word_vec_size,
            word_vocab_size=self.word_vocab_size,
            word_padding_idx=self.word_padding_idx
        )
        self.max_relative_positions = 0
        self.dropout = 0.3
        self.attn_dropout = 0.1
        self.lr = 0.001
        self.mometum = 0.9
        self.num_epoch = 20
        self.batch_size = 32
        self.copy_attn = False
        self.self_attn_type = 'scaled_dot'
        self.aan_useffn = True
        self.full_ctxt_alignt = True
        self.aan_useffn = False
        self.full_context_alignment = True
        self.alignment_layer = 1
        self.alignment_heads = 1


def generate_trainer() -> tuple[model.NMTModel, loss.LossCompute, optim.SGD, trainer.Trainer]:
    conf = Config()
    tf_enc = encoder.TransformerEncoder(d_model=conf.input_size,
                                        heads=conf.heads,
                                        d_ff=conf.d_ff,
                                        dropout=conf.dropout,
                                        attention_dropout=conf.attn_dropout,
                                        num_layers=conf.num_layers,
                                        embeddings=conf.embeddings,
                                        max_relative_positions=conf.max_relative_positions)

    tf_dec = decoder.TransformerDecoder(num_layers=conf.num_layers,
                                        d_model=conf.input_size,
                                        heads=conf.heads,
                                        d_ff=conf.d_ff,
                                        copy_attn=conf.copy_attn,
                                        self_attn_type=conf.self_attn_type,
                                        dropout=conf.dropout,
                                        attention_dropout=conf.attn_dropout,
                                        embeddings=conf.embeddings,
                                        max_relative_positions=conf.max_relative_positions,
                                        aan_useffn=conf.aan_useffn,
                                        full_context_alignment=conf.full_context_alignment,
                                        alignment_layer=conf.alignment_layer,
                                        alignment_heads=conf.alignment_heads
                                        )

    net = model.NMTModel(encoder=tf_enc, decoder=tf_dec)

    loss_fn = loss.LossCompute(criterion=nn.CrossEntropyLoss,
                                generator=net) # seems to be many other losses

    # opts.config_opts(net) #supposed to create an opt (revoir)
    
    optimizer = optim.SGD(net.parameters(), lr=conf.lr, momentum=conf.mometum)
    
    parser = tr._get_parser()
    opt, unknown = parser.parse_known_args()
    ArgumentParser.update_model_opts(opt)
    ArgumentParser.validate_model_opts(opt)

    opt.dump_samples = False
    #opt.data.items = []
    print(f"opt : {opt}")
    transforms = opt.transforms    
    print(f"DO")
    vocab = bld_voc.build_vocab(opt, transforms)
    print(f"DONE")

    scorer_prep = ScoringPreparator(vocabs=vocab, opt=opt) 

    train_scorer = {} # keeps in memory the values of the training metrics
    valid_scorer = {} # keeps in memory the values of the validation metrics


    coach = trainer.Trainer(model=net,
                            train_loss=loss_fn,
                            valid_loss=loss_fn,
                            scoring_preparator=scorer_prep,
                            train_scorers=train_scorer,
                            valid_scorers=valid_scorer,
                            optim=optimizer
                            )

    return net,loss_fn, optimizer, coach

def training() -> None:
    # training_iter = _build_train_iter(net.opt)
    # training_steps = 20
    # net.train(train_iter= training_iter,
    #           train_steps= training_steps)
    
    #sgl.main(opt=net.opt,fields=,transforms_cls=)
    
    parser = tr._get_parser()
    opt, unknown = parser.parse_known_args()
    tr.train(opt)
    





if __name__ == '__main__':
    print(f"\n----------------------- Beginning of script -----------------------\n")
    # print(f"Generating model")
    generate_trainer()
    # print(f"Model generated")
    print(f"Training model")
    training()
    print(f"Model trained")

