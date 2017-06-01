#!/bin/bash

if [ ! -d "data/glove" ]; then
  mkdir -p "data/glove"
fi

cd data/glove
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
rm glove.6B.zip
cd $OLDPWD

SRLPATH="./data/srl"
if [ ! -d $SRLPATH ]; then
  mkdir -p $SRLPATH
fi

# Get srl-conll package.
wget -O "${SRLPATH}/srlconll-1.1.tgz" http://www.lsi.upc.edu/~srlconll/srlconll-1.1.tgz
tar xf "${SRLPATH}/srlconll-1.1.tgz" -C "${SRLPATH}"
rm "${SRLPATH}/srlconll-1.1.tgz"

