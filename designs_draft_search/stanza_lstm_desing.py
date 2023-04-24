import numpy as np
import matplotlib.pyplot as plt

import sys

import torch 
import torch.nn as nn
from collections import OrderedDict

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim



# Print the informations of the model
PATH = '/home/zk274707/utils-git/stanza-multilingual/models/langid/ud.pt'
ld = torch.load(PATH, map_location=torch.device('cpu'))

print("Model's parameters: \n")

print('number of layers : ', ld['num_layers'])
print('embeding dimensions : ',ld['embedding_dim'])
print('hiden dimensions :', ld['hidden_dim'])

print('\n--------------------------------------------- \n')

print("Model's state_dict: \n")
for param_tensor in ld['model_state_dict']:
    print(param_tensor)

# Define model parameters
num_layers = ld['num_layers']
embedding_dim = ld['embedding_dim']
hidden_dim = ld['hidden_dim']

# Load state dictionary
state_dict = ld['model_state_dict']

# Define model architecture
model = torch.nn.Sequential( OrderedDict([
    ('loss_train', torch.nn.Embedding(num_embeddings=state_dict['char_embeds.weight'].shape[0],
                       embedding_dim=embedding_dim)),
    ('lstm',torch.nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,
                  num_layers=num_layers, bidirectional=True)),
    ('hidden_to_tag',torch.nn.Linear(in_features=hidden_dim*2, out_features=state_dict['hidden_to_tag.weight'].shape[0]))
])
)

print(state_dict.keys())
print('\n--------------------------------------------- \n')
print(model.state_dict().keys())

print(state_dict.keys() == model.state_dict().keys())
# # Load state dictionary into model
# model.load_state_dict(state_dict)

# doc = model('How are you ?')
# print(doc)