#!/bin/bash

help(){
    echo "Usage onmt_training - it aims to build the dataset and the vocab before training the model according to the config file
            [-t | --train] : path to the .conllu training file
            [-d | --dev] : path to the .conllu validation file
            [-c | --config] : config file .yaml format used for the training
            [-f | --fields] : fields of the conllu file to be used for validation, can use multiple ones separated by '-' -> upos /xpos / feats / head / deprel / deps
            [-w | --write] : write in the config file if it is given -> true/false
            [-u | --use] : choose to build the dataset according to the cli or the config file -> cli / config
    "
    exit 2
}

# Default values
TRAIN=""
DEV=""
CONFIG=""
FIELDS=""
WRITE=""
USE=""

SHORT=t:,d:,c:,f:,w:,u:,h
LONG=train:,dev:,config:,fields:,write:,use:,help
OPTS=$(getopts -a -n onmt_training --options $SHORT --longoptions $LONG -- "$@" opt)

OPTS=${OPTS::-2}
# Parse options
while getopts "t:d:c:f:w:u:h" "train:,dev:,config:,fields:,write:,use:,help" opt; do
  case $opt in
    t) TRAIN="$OPTARG";;
    d) DEV="$OPTARG";;
    c) CONFIG="$OPTARG";;
    f) FIELDS="$OPTARG";;
    w) WRITE="$OPTARG";;
    u) USE="$OPTARG";;
    \?) echo "Invalid option: -$OPTARG" >&2;help; exit 1;;
  esac
done

# Long options
for arg in "$@"; do
  case $arg in
    --train*) TRAIN="${arg#--train}";;
    --dev*) DEV="${arg#--dev}";;
    --config*) CONFIG="${arg#--config}";;
    --fields*) FIELDS="${arg#--fields}";;
    --write*) WRITE="${arg#--write}";;
    --use*) USE="${arg#--use}";;
  esac
done

echo "TRAIN: $TRAIN"
echo "DEV: $DEV"
echo "CONFIG: $CONFIG"
echo "FIELDS: $FIELDS"
echo "WRITE: $WRITE"
echo "USE: $USE"
