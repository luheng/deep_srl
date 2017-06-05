#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda:/usr/local/cuda/lib64:/opt/OpenBLAS/lib

MODEL_PATH="conll05_propid_model"
MODEL_PATH2="conll05_ensemble"
TEMP_PATH="temp"

INPUT_PATH=$1
OUTPUT_PATH=$2

TH_FLAGS="optimizer=fast_compile,floatX=float32"
if [ "$#" -gt 2 ]
then
  TH_FLAGS="${TH_FLAGS},device=gpu$3,lib.cnmem=0.9"
fi

THEANO_FLAGS=$TH_FLAGS python python/predict.py \
  --model=$MODEL_PATH \
  --task="propid" \
  --input="${INPUT_PATH}" \
  --output="${TEMP_PATH}/sample.pp.txt" \
  --outputprops="${TEMP_PATH}/sample.pred.props" 

THEANO_FLAGS=$TH_FLAGS python python/predict.py \
  --model=$MODEL_PATH2 \
  --task="srl" \
  --input="${TEMP_PATH}/sample.pp.txt" \
  --inputprops="${TEMP_PATH}/sample.pred.props" \
  --output="${OUTPUT_PATH}"


