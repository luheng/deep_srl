#! /bin/bash

SRLPATH="./data/srl"

if [ ! -d $SRLPATH ]; then
  mkdir -p $SRLPATH
fi

export PERL5LIB="$SRLPATH/srlconll-1.1/lib:$PERL5LIB"
export PATH="$SRLPATH/srlconll-1.1/bin:$PATH"

WSJPATH=$1

TRAIN_SECTIONS=(02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21)
DEVEL_SECTIONS=(24)

# Fetch data
wget -O "${SRLPATH}/conll05st-release.tar.gz" http://www.lsi.upc.edu/~srlconll/conll05st-release.tar.gz
wget -O "${SRLPATH}/conll05st-tests.tar.gz" http://www.lsi.upc.edu/~srlconll/conll05st-tests.tar.gz
tar xf "${SRLPATH}/conll05st-release.tar.gz" -C "${SRLPATH}"
tar xf "${SRLPATH}/conll05st-tests.tar.gz" -C "${SRLPATH}"

CONLL05_PATH="${SRLPATH}/conll05st-release"

if [ ! -d "${CONLL05_PATH}/train/words" ]; then
  mkdir -p "${CONLL05_PATH}/train/words"
fi

if [ ! -d "${CONLL05_PATH}/devel/words" ]; then
  mkdir -p "${CONLL05_PATH}/devel/words"
fi

# Retrieve words from PTB source.
for s in "${TRAIN_SECTIONS[@]}"
do
  echo $s
  cat ${WSJPATH}/parsed/mrg/wsj/$s/* | wsj-removetraces.pl | wsj-to-se.pl -w 1 | awk '{print $1}' | \
    gzip > "${CONLL05_PATH}/train/words/train.$s.words.gz"
done

for s in "${DEVEL_SECTIONS[@]}"
do
  echo $s
  cat ${WSJPATH}/parsed/mrg/wsj/$s/* | wsj-removetraces.pl | wsj-to-se.pl -w 1 | awk '{print $1}' | \
    gzip > "${CONLL05_PATH}/devel/words/devel.$s.words.gz"
done

rm "${SRLPATH}/conll05st-release.tar.gz"
rm "${SRLPATH}/conll05st-tests.tar.gz"

cd ${CONLL05_PATH}
./scripts/make-trainset.sh
./scripts/make-devset.sh

# Prepare test set.
zcat test.wsj/words/test.wsj.words.gz > /tmp/words
zcat test.wsj/props/test.wsj.props.gz > /tmp/props
paste -d ' ' /tmp/words /tmp/props  > "test-wsj"
echo Cleaning files
rm -f /tmp/$$*

zcat test.brown/words/test.brown.words.gz > /tmp/words
zcat test.brown/props/test.brown.props.gz > /tmp/props
paste -d ' ' /tmp/words /tmp/props  > "test-brown"
echo Cleaning files
rm -f /tmp/$$*

cd $OLDPWD

# Process CoNLL05 data
zcat "${CONLL05_PATH}/devel/props/devel.24.props.gz" > "${SRLPATH}/conll05.devel.props.gold.txt"
zcat "${CONLL05_PATH}/test.wsj/props/test.wsj.props.gz" > "${SRLPATH}/conll05.test.wsj.props.gold.txt"
zcat "${CONLL05_PATH}/test.brown/props/test.brown.props.gz" > "${SRLPATH}/conll05.test.brown.props.gold.txt"

zcat "${CONLL05_PATH}/train-set.gz" > "${CONLL05_PATH}/train-set"
zcat "${CONLL05_PATH}/dev-set.gz" > "${CONLL05_PATH}/dev-set"

# Convert CoNLL format to seq2seq.
python preprocess/process_conll05.py "${CONLL05_PATH}/train-set" "${SRLPATH}/conll05.train.txt" \
  "${SRLPATH}/conll05.propid.train.txt" 5
python preprocess/process_conll05.py "${CONLL05_PATH}/dev-set" "${SRLPATH}/conll05.devel.txt" \
  "${SRLPATH}/conll05.propid.devel.txt" 5
python preprocess/process_conll05.py "${CONLL05_PATH}/test-wsj" "${SRLPATH}/conll05.test.wsj.txt" \
  "${SRLPATH}/conll05.propid.test.wsj.txt" 1
python preprocess/process_conll05.py "${CONLL05_PATH}/test-brown" "${SRLPATH}/conll05.test.brown.txt" \
  "${SRLPATH}/conll05.propid.test.brown.txt" 1


