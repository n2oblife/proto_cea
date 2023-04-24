import sys
sys.path.append('/home/zk274707/Projet/protos')

import torch
import torch.nn as nn
from proto.tok_and_DM import tok_and_DM

try :
    model = tok_and_DM(64)
    output = model("Cette phrase est un test.")
    print(output)

except:
    KeyboardInterrupt()

finally:
    print("\n"+"End of training script")