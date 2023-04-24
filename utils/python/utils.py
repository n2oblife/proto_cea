import numpy as np
import matplotlib.pyplot as plt

import sys

import torch 
import torch.nn as nn
from collections import OrderedDict

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim

def get_var_from_memory(var : any):
    '''Not always working'''
    # Get the memory address of the variable
    var_address = id(var)

    # Get the namespace where the variable is defined
    for ns in globals(), locals():
        if var_address in ns.values():
            namespace = ns
            break

    # Access the variable using its name in the namespace
    return namespace["var"]