from typing import Any
import torch
from transformers import AutoTokenizer, NllbTokenizer
from transformers.adapters import XLMRobertaAdapterModel, AdapterConfig

class Tok_xlmR(nn.Module):
    def __init__(self) -> None:
        '''This class is composed of NllbTokenizer from facebook and XLMRoberta 
        which supports adapters.'''
        super(Tok_xlmR, self).__init__()
        self._tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        self._model = XLMRobertaAdapterModel.from_pretrained("xlm-roberta-base")
        self._model_name = 'xlm-roberta-base'
    
    def __call__(self, inputs : str | list[str], *args: Any, **kwds: Any) -> torch.tensor:
        outputs = self._tokenizer(inputs, return_tensors='pt')
        outputs = self._model(**outputs, return_tensors='pt')
        return outputs
    
    def add_adapter(self, task : str = 'Dep', lgge : str = 'English') -> None:
        self._adapter_config = AdapterConfig.load("pfeiffer") 
        self._task = task
        self._lgge = lgge

        self._adapter_name = 'deeplima_adapters_'+task+'_'+lgge
        self._model.add_adapter(self._adapter_name)

    def add_classification_head(self, adapter_name : str, num_labels : int, id2label : dict) -> None :
        self._model.add_classification_head(adapter_name,num_labels, id2label)
    
    def set_active_adapters(self, adapter_name : str) -> None:
        self._model.train()
        self._model.set_active_adapters(adapter_name)
    
    def has_adapters(self) -> bool:
        return self._model.has_adapters()
    
    def save_pretrained(self, path : str) -> None:
        self._model.save_pretrained(path)
    
    def save_adapter(self, path : str) -> None :
        self._model.save_adapter(path)
