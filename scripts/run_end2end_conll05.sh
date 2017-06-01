#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda:/usr/local/cuda/lib64:/opt/OpenBLAS/lib

MODEL_PATH="./conll05_propid_model"
MODEL_PATH2="./conll05_ensemble"
TEMP_PATH="./temp"

TH_FLAGS="optimizer=fast_compile,floatX=float32"
if [ "$#" -gt 0 ]
then
  TH_FLAGS="${TH_FLAGS},device=gpu$1,lib.cnmem=0.9"
fi


THEANO_FLAGS=$TH_FLAGS python python/predict.py \
  --model=$MODEL_PATH \
  --task="propid" \
  --input="./data/srl/conll05.propid.devel.txt" \
  --output="$TEMP_PATH/conll05.devel.pp.txt" \
  --outputprops="$TEMP_PATH/conll05.devel.props.pred.txt" \
  --gold="./data/srl/conll05.devel.props.gold.txt" 

THEANO_FLAGS=$TH_FLAGS python python/predict.py \
  --model=$MODEL_PATH2 \
  --task="srl" \
  --input="$TEMP_PATH/conll05.devel.pp.txt" \
  --gold="./data/srl/conll05.devel.props.gold.txt" \
  --inputprops="$TEMP_PATH/conll05.devel.props.pred.txt" 


