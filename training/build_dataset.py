import pyconll
from transformers import XLMRobertaTokenizer


# build dataset for onmt (v1 - v2 with spaces between tokens)

tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

tgt_conllu_path = '/home/zk274707/Projet/datasets/ud-treebanks-v2.10-trainable/UD_English-EWT/en_ewt-ud-train.conllu'
tgt_conllu_file = pyconll.load_from_file(tgt_conllu_path)

src_txt_path = '/home/zk274707/Projet/datasets/ud-treebanks-v2.10-trainable/UD_English-EWT/en_ewt-ud-train_src_onmt.txt'
tgt_txt_path = '/home/zk274707/Projet/datasets/ud-treebanks-v2.10-trainable/UD_English-EWT/en_ewt-ud-train_tgt_onmt.txt'
src_file = open(src_txt_path,'a')
tgt_file = open(tgt_txt_path, 'a')

line_tokens, line_deprel = '', ''

for sentence in tgt_conllu_file :
    print(f"this is a sentence : {sentence.text}")
    for token in sentence :
        print(f"this is a token : {token.form} -> deprel : {token.deprel} + deps : {token.deps}")
        line_tokens = line_tokens + ' ' + str(token)
        line_deprel = line_deprel + ' ' + str(token.deprel) #either deprel or deps (enhanced with head)
    src_file.write(line_tokens + '\n')
    tgt_file.write(line_deprel + '\n')
    line_tokens, line_deprel = '', ''

src_file.close()
tgt_file.close()