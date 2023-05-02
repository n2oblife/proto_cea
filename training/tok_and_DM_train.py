import sys
sys.path.insert(1, '/home/zk274707/Projet/proto/proto')
import torch
import torch.nn as nn
from tok_and_DM import tok_and_DM

try :
    hidden_dim = 64
    dep_labels = 37
    model = tok_and_DM(input_hidden_arc_dim=hidden_dim, input_out_dim=dep_labels)
    batch_sentences = ["Hello I'm a single sentence",
                    "And another sentence",
                    "And the very very last one"]
    output = model(batch_sentences)
    print(output)

except:
    KeyboardInterrupt()

finally:
    print("\n"+"End of training script")