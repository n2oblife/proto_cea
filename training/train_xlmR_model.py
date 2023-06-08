import torch
import sys
if torch.cuda.is_available():
    # FactotyAI
    sys.path.append('/home/users/zkanit/')
else :
    # PC
    sys.path.append('/home/zk274707/Projet/proto/')

from xlmr_model import *
from utils.python.utils import Device, load_seed
device = Device() # adapt paths according to the device used
load_seed()





if __name__ == '__main__':
    