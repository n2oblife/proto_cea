import torch 
import torch.nn as nn
from collections import OrderedDict
import stanza 

nlp = stanza.Pipeline('en') # download th English model and initialize an English neural pipeline
doc = nlp("Barack Obama was born in Hawaii.  He was elected president in 2008.")
doc.sentences[0].print_dependencies()

# print('\n--------------------------------------------- \n')
# print(nlp)
# print('\n--------------------------------------------- \n')


# PATH = '/home/zk274707/stanza_resources/en/depparse/combined.pt'
# ld = torch.load(PATH)
# print(ld)

print(nlp.processors['depparse']._trainer.model)
