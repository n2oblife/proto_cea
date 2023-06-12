#!/usr/bin/env bash

n_cpus= lscpu |grep 'CPU(s)' | awk '{print $2;}' | head -n 1

onmt_build_vocab -config onmt_config.yaml -n_sample -1  -num_threads $n_cpus

onmt_train -config onmt_config.yaml