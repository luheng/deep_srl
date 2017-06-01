#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda:/usr/local/cuda/lib64:/opt/OpenBLAS/lib

MODEL_PATH="./conll05_ensemble"
INPUT_PATH="data/srl/conll05.devel.txt"
GOLD_PATH="data/srl/conll05.devel.props.gold.txt"
OUTPUT_PATH="temp/conll05.devel.out"

if [ "$#" -gt 0 ]
then
THEANO_FLAGS="optimizer=fast_compile,device=gpu$1,floatX=float32,lib.cnmem=0.8" python python/predict.py \
  --model="$MODEL_PATH" \
  --input="$INPUT_PATH" \
  --output="$OUTPUT_PATH" \
  --gold="$GOLD_PATH"
else
THEANO_FLAGS="optimizer=fast_compile,floatX=float32" python python/predict.py \
  --model="$MODEL_PATH" \
  --input="$INPUT_PATH" \
  --output="$OUTPUT_PATH" \
  --gold="$GOLD_PATH"
fi

