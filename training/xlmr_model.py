from typing import Any
import torch
import sys
if torch.cuda.is_available():
    # FactotyAI
    sys.path.append('/home/users/zkanit/')
else :
    # PC
    sys.path.append('/home/zk274707/Projet/proto/')

from transformers import AutoTokenizer, NllbTokenizer
from transformers.adapters import XLMRobertaAdapterModel, AdapterConfig
from var_utils import *
from utils.python.utils import Device
device = Device() # adapt paths according to the device used

# ----------------------------------
tokenizer_nllb = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
tokenizer_xlm = AutoTokenizer.from_pretrained("xlm-roberta-base")

inputs_nllb = tokenizer_nllb("Hello, my dog is cute", return_tensors="pt")
inputs_xlm = tokenizer_xlm("Hello, my dog is cute", return_tensors="pt")

print(f"inputs_nllb : {inputs_nllb}")
print(f"inputs_xlm : {inputs_xlm}")

# model = XLMRobertaAdapterModel.from_pretrained("xlm-roberta-base")
# ----------------------------------



class Tok_xlmR(nn.Module):
    def __init__(self) -> None:
        '''This class is composed of NllbTokenizer from facebook and XLMRoberta 
        which supports adapters.'''
        super(Tok_xlmR, self).__init__()
        self._tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        self._model = XLMRobertaAdapterModel.from_pretrained("xlm-roberta-base")
        self._model_name = 'xlm-roberta-base with adapters'
        self._adapter_config = AdapterConfig.load("pfeiffer")
    
    def __call__(self, inputs : Union[str, list[str]], *args: Any, **kwds: Any) -> torch.tensor:
        outputs = self._tokenizer(inputs, return_tensors='pt')
        outputs = self._model(**outputs, return_tensors='pt')
        return outputs
    
    def add_adapter(self, task : str = 'Dep', lgge : str = 'English') -> None:
        assert task in TASKS, f"task must be {TASKS}"
        self._task = task
        assert lgge in LGGE, f"language must be {LGGE}"
        self._lgge = lgge

        self._adapter_name = 'deeplima_adapters_'+task+'_'+lgge
        self._model.add_adapter(self._adapter_name)

    def add_classification_head(self, adapter_name : str, num_labels : int, id2label : dict) -> None :
        self._model.add_classification_head(adapter_name,num_labels, id2label)
    
    def set_active_adapters(self, adapter_name : str) -> None:
        """Activates the adapters before using or training them, multiple adapters
        can be activated on top of each others

        Args:
            adapter_name (str): name of the adapter to activate
        """
        self._model.train()
        self._model.set_active_adapters(adapter_name)
    
    def has_adapters(self) -> bool:
        return self._model.has_adapters()
    
    def save_pretrained(self, path : str) -> None:
        """Saves the whole model in the path as a .pt from the AdapterHub lib

        Args:
            path (str): path where to save model
        """
        self._model.save_pretrained(path)
    
    def save_adapter(self, path : str) -> None :
        """Saves the adapters in the path as a .pt from the AdapterHub lib

        Args:
            path (str): path where to save model
        """
        self._model.save_adapter(path)