#! /bin/bash

SRLPATH="./data/srl"
ONTONOTES_PATH=$1

if [ ! -d $SRLPATH ]; then
  mkdir -p $SRLPATH
fi

python preprocess/process_conll2012.py \
  "${ONTONOTES_PATH}/data/train/data/english/annotations/" \
  "${SRLPATH}/conll2012.train.txt" \
  "${SRLPATH}/conll2012.train.props.gold.txt" \
  "${SRLPATH}/conll2012.propid.train.txt" \
  "${SRLPATH}/conll2012.train.domains"

python preprocess/process_conll2012.py \
  "${ONTONOTES_PATH}/data/development/data/english/annotations/" \
  "${SRLPATH}/conll2012.devel.txt" \
  "${SRLPATH}/conll2012.devel.props.gold.txt" \
  "${SRLPATH}/conll2012.propid.devel.txt" \
  "${SRLPATH}/conll2012.devel.domains"

python preprocess/process_conll2012.py \
  "${ONTONOTES_PATH}/data/conll-2012-test/data/english/annotations/" \
  "${SRLPATH}/conll2012.test.txt" \
  "${SRLPATH}/conll2012.test.props.gold.txt" \
  "${SRLPATH}/conll2012.propid.test.txt" \
  "${SRLPATH}/conll2012.test.domains"



