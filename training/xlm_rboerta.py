import torch
import sys

if torch.cuda.is_available():
    # FactotyAI
    sys.path.apend('/home/users/zkanit/')
else :
    # PC
    sys.path.apend('/home/zk274707/Projet/proto/')


from transformers.adapters import XLMRobertaAdapterModel, AutoAdapterModel, BertAdapterModel, AdapterConfig

from utils.python.utils import Device, load_seed
device = Device() # adapt paths according to the device used
load_seed()




if __name__ == '__main__':
    