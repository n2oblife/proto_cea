# import inspect


# import torch
# from trankit.tpipeline import TPipeline
# import onmt.decoders as dec
# from transformers.adapters import XLMRobertaAdapterModel

# model = XLMRobertaAdapterModel.from_pretrained('xlm-roberta-base')
# path = '/home/zk274707/Projet/proto_utils/save_dir/xlm-roberta-base/xlm-roberta-base.pt'
# torch.save(model, path)

# config={
#     'category': 'customized', # pipeline category
#     'task': 'posdep', # task name
#     'save_dir': '../proto_utils/save_dir', # directory for saving trained model
#     'gpu' : torch.cuda.is_available(),
#     'max_epoch':150,
#     'train_conllu_fpath': '/home/zk274707/Projet/datasets/ud-treebanks-v2.10-trainable/UD_English-EWT/en_ewt-ud-train.conllu',
#     'dev_conllu_fpath': '/home/zk274707/Projet/datasets/ud-treebanks-v2.10-trainable/UD_English-EWT/en_ewt-ud-dev.conllu'
#     }
# model = TPipeline(config) 

# deco = dec.DecoderBase()

# '''
# attributes = inspect.getmembers(MyClass, lambda a:not(inspect.isroutine(a)))
# print([a for a in attributes if not(a[0].startswith('__') and a[0].endswith('__'))])
# '''


# print("TPipelinne : ")
# print(model.__dict__.keys())
# print('-----------------------')
# print("DecoderBase : ")
# print(deco.__dict__.keys())

# ['_lang', '_task', '_train_txt_fpath', '_train_conllu_fpath', '_dev_txt_fpath', '_dev_conllu_fpath', '_train_bio_fpath', '_dev_bio_fpath', '_text_split_by_space', '_save_dir', '_cache_dir', '_gpu', '_use_gpu', '_ud_eval', '_config', 'logger', 'train_set', 'batch_num', 'dev_set', 'dev_batch_num', '_embedding_layers', '_tagger', 'model_parameters', 'optimizer', 'schedule']


# ['training', '_parameters', '_buffers', '_non_persistent_buffers_set', '_backward_hooks', '_is_full_backward_hook', '_forward_hooks', '_forward_pre_hooks', '_state_dict_hooks', '_load_state_dict_pre_hooks', '_load_state_dict_post_hooks', '_modules', 'attentional']


from ctranslate2.specs import attention_spec, common_spec, model_spec, posdep_spec, xlmr_spec, transformer_spec

TASK = ['tokenize', 'posdep', 'mwt', 'lemmatize', 'ner']
NEED_EMBEDDING = ['tokenize', 'posdep', 'ner']

EMBEDDING_CLS = {'xlmr': xlmr_spec.XLMRobertaModelSpec} # encoder
TASK_CLS = {'posdep': posdep_spec.PosdepDecoderSpec} # decoder

class MyClass:
    def __init__(self, param1, param2, param3):
        self.param1 = param1
        self.param2 = param2
        self.param3 = param3

class PosdepDecoderSpec(model_spec.LayerSpec):
    # TODO be sure that during conversion the forward keeps the same
    def __init__(self) -> None:
        self.upos_embeddings = None
        # postagging
        self.upos_ffn = MyClass(10, 'hello', True)
        self.xpos_ffn = None
        self.feats_ffn = None
        self.down_project = None
        # dep parsing
        self.unlabeled = None
        self.deprel = None


# Create an instance of the class
my_instance = MyClass(common_spec.LinearSpec(), 'hello', True)
pos = PosdepDecoderSpec()
# Get the attribute names of the instance

class BertOutputAdaptersMixinSpec(model_spec.LayerSpec):
    def __init__(self) -> None:
        super().__init__()

class BertLayerNormSpec(common_spec.LayerNormSpec):
    """DONE"""
    def __init__(self) -> None:
        """Nothing else than a layer norm but trankit wanted it to be different so we keep the architecture"""
        super().__init__()
        # TODO pb of the epsilon not managed during conversion
        self

class BertOutputSpec(model_spec.LayerSpec, BertOutputAdaptersMixinSpec):
    def __init__(self) -> None:
        super().__init__()
        self.dense = common_spec.LinearSpec()
        self.layer_norm = BertLayerNormSpec()
        # TODO complete the adaptermixin implementation