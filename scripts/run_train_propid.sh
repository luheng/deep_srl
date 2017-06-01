export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda:/usr/local/cuda/lib64:/opt/OpenBLAS/lib

CONFIG="config/propid_config.json"
LOGDIR="conll05_propid_model_l2_h300"
TRAIN_PATH="data/srl/conll05.propid.train.txt"
DEV_PATH="data/srl/conll05.propid.devel.txt"

THEANO_FLAGS="mode=FAST_RUN,device=gpu$1,floatX=float32,lib.cnmem=0.8" python python/train.py \
  --config=$CONFIG \
   --model=$LOGDIR \
   --task='propid' \
   --train=$TRAIN_PATH \
   --dev=$DEV_PATH 
