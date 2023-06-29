import pyconll
import sys

# There might be a problem with .../OpenNMT-py/onmt/utils/parse.py", line 30 when building vocab (correction useless ?)

if __name__=='__main__':
    # build dataset for onmt (v1 - v2 with spaces between tokens)

    dataset_type = str(sys.argv[1])
    print(dataset_type)
    assert dataset_type in ['train', 'dev'], "This script needs an argument, the dataset type must be train or dev"

    # TODO give the whole path to adapt for the whole UD dataset
    tgt_conllu_path = '/home/zk274707/Projet/datasets/ud-treebanks-v2.10-trainable/UD_English-EWT/en_ewt-ud-'+dataset_type+'.conllu'
    tgt_conllu_file = pyconll.load_from_file(tgt_conllu_path)

    src_txt_path = '/home/zk274707/Projet/datasets/ud-treebanks-v2.10-trainable/UD_English-EWT/en_ewt-ud-'+dataset_type+'_src_onmt.txt'
    tgt_txt_path = '/home/zk274707/Projet/datasets/ud-treebanks-v2.10-trainable/UD_English-EWT/en_ewt-ud-'+dataset_type+'_tgt_onmt.txt'
    src_file = open(src_txt_path,'a')
    tgt_file = open(tgt_txt_path, 'a')

    line_tokens, line_deprel = '', ''

    for sentence in tgt_conllu_file :
        print(f"this is a sentence : {sentence.text}")
        for token in sentence :
            print(f"this is a token : {token.form} -> deprel : {token.deprel} + deps : {token.deps}")
            # deals with the multi token words
            if not(token.deprel == None):
                line_tokens += str(token.form) + ' '
                line_deprel += str(token.deprel)  + ' ' #either deprel or deps (enhanced with head)
        src_file.write(line_tokens + '\n')
        tgt_file.write(line_deprel + '\n')
        line_tokens, line_deprel = '', ''

    src_file.close()
    tgt_file.close()