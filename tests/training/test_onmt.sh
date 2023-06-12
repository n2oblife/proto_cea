#!/usr/bin/env bash

cd proto_utils/datasets/ud-treebanks-v2.10-trainable/UD_English-EWT/

n_cpus= lscpu |grep 'CPU(s)' | awk '{print $2;}' | head -n 1
onmt_build_vocab -config onmt_config.yaml -n_sample -1  -num_threads $n_cpus


# python tests/training/test_onmt.py --data ~/tutos/translation_WMT17/OpenNMT-py/data/data.yaml --src_vocab ~/tutos/translation_WMT17/OpenNMT-py/docs/source/examples/wmt17_en_de/vocab.shared -share_vocab
onmt_train -config onmt_config.yaml