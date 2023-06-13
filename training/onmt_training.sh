#!/usr/bin/env bash

# getting the variables
# file_path= ...
# n_lines=wc -l $file |awk '{print $1;}'
n_cpus= lscpu |grep 'CPU(s)' | awk '{print $2;}' | head -n 1

# TODO : multiply for hyper threading
# threads_per_cpu= lscpu |grep 'Thread(s) per core:' |awk '{print $4;}'
# n_threads=$ expr $n_cpus '*' $threads_per_cpu

onmt_build_vocab -config onmt_config.yaml -n_sample -1  -num_threads $n_cpus

onmt_train -config onmt_config.yaml 