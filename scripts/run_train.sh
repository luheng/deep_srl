export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda:/usr/local/cuda/lib64:/opt/OpenBLAS/lib

CONFIG="config/srl_small_config.json"
MODEL="conll05_small_model"
TRAIN_PATH="data/srl/conll05.train.txt"
DEV_PATH="data/srl/conll05.devel.txt"
GOLD_PATH="data/srl/conll05.devel.props.gold.txt"

THEANO_FLAGS="mode=FAST_RUN,device=gpu$1,floatX=float32,lib.cnmem=0.9" python python/train.py \
   --config=$CONFIG \
   --model=$MODEL \
   --train=$TRAIN_PATH \
   --dev=$DEV_PATH \
   --gold=$GOLD_PATH
