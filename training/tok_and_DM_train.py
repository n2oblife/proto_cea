import torch
import torch.nn as nn
from proto.tok_and_DM import tok_and_DM

try :
    model = tok_and_DM(64)
    batch_sentences = ["Hello I'm a single sentence",
                    "And another sentence",
                    "And the very very last one"]
    output = model(batch_sentences)
    print(output)

except:
    KeyboardInterrupt()

finally:
    print("\n"+"End of training script")