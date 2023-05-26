import torch
import torch.nn as nn
import torch.optim as optim

import onmt
import onmt.models.model as model
import onmt.encoders as encoder
import onmt.decoders as decoder
import onmt.trainer as trainer
import onmt.utils.loss as loss
import onmt.translate.utils as tsl_util

seed  = 2147483647
torch.manual_seed(seed)

class Config():
    def __init__(self) -> None:
        self.input_size = 1024
        self.heads = 3
        self.d_ff = 2
        self.dropout = 0.3
        self.lr = 0.001
        self.mometum = 0.9
        self.num_epoch = 20
        self.batch_size = 64

def generate_trainer() -> :
    conf = Config()
    tf_enc = encoder.TransformerEncoder(d_model=conf.input_size,
                                        heads=conf.heads,
                                        d_ff=conf.d_ff,
                                        dropout=conf.dropout,
                                        attention_dropout=conf.dropout)

    tf_dec = decoder.TransformerDecoder(d_model=conf.input_size,
                                        heads=conf.heads,
                                        d_ff=conf.d_ff,
                                        dropout=conf.dropout,
                                        attention_dropout=conf.dropout)

    net = model.NMTModel(encoder=tf_enc, decoder=tf_dec)

    loss_fn = loss.LossComputeBase(criterion=nn.CrossEntropyLoss,
                                generator=net) # seems to be many other losses

    scorer_prep = tsl_util.ScoringPreparator() # TODO : finir le constructeur

    train_scorer = {} # keeps in memory the values of the training metrics
    valid_scorer = {} # keeps in memory the values of the validation metrics

    optimizer = optim.SGD(net.parameters(), lr=conf.lr, momentum=conf.mometum)

    coach = trainer.Trainer(model=net,
                            train_loss=loss_fn,
                            valid_loss=loss_fn,
                            scoring_preparator=scorer_prep,
                            train_scorers=train_scorer,
                            valid_scorers=valid_scorer,
                            optim=optimizer
                            )
    return net,loss_fn, optimizer, coach

def training(net : ):
    opt = net.ge
    loader = 
    for batch in loader:
        score = trainer.training_eval_handler()
