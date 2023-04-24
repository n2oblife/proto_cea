import torch
from transformers import NllbTokenizer
from trankit import Pipeline, TPipeline


tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

print(tokenizer.model_max_length)

my_var = NllbTokenizer.max_len_single_sentence


# # Get the memory address of the variable
# my_var_address = id(my_var)

# # Get the namespace where the variable is defined
# for ns in globals(), locals():
#     if my_var_address in ns.values():
#         my_namespace = ns
#         break

# # Access the variable using its name in the namespace
# print(vars(tokenizer))

# print(NllbTokenizer.max_len_single_sentence)
# print(NllbTokenizer.max_len_sentences_pair)
# for param in tokenizer.parameters:
#     print(param)

batch_sentences = ["Hello I'm a single sentence",
                   "And another sentence",
                   "And the very very last one"]
doc = tokenizer(batch_sentences, padding = 'max_length')

print(doc)

# training_config={
#     'category': 'customized-mwt-ner', # pipeline category
#     'task': 'posdep', # task name
#     'save_dir': './save_dir', # directory for saving trained model
#     'train_conllu_fpath': './train.conllu', # annotations file in CONLLU format  for training
#     'dev_conllu_fpath': './dev.conllu' # annotations file in CONLLU format for development
#     }

# # initialize a trainer for the task
# trainer = TPipeline(training_config)

# print(trainer)

# treebank = "auto"
# pipe = Pipeline(lang=treebank)
# print(pipe)
# pipe._setup_config(lang=treebank)
# conf = pipe._config
# print(conf)
# classifier = Pipeline.posdep(config=conf, treebank_name=treebank)
# print(classifier)