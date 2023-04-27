from transformers import NllbTokenizer
import torch

tok = NllbTokenizer.from_pretrained('facebook/nllb-200-distilled-600M')
path = './save_dir/HG/nllb_tok.pt'
torch.save(tok, path)