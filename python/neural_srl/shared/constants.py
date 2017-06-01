from os.path import join
import os
import random

ROOT_DIR = join(os.path.dirname(os.path.abspath(__file__)), '../../../')

RANDOM_SEED = 12345
random.seed(RANDOM_SEED)

SRL_CONLL_EVAL_SCRIPT  = join(ROOT_DIR, 'scripts/run_eval.sh')
WORD_EMBEDDINGS = { "glove50": join(ROOT_DIR, 'data/glove/glove.6B.50d.txt'),
                    "glove100": join(ROOT_DIR, 'data/glove/glove.6B.100d.txt'),
                    "glove200": join(ROOT_DIR, 'data/glove/glove.6B.200d.txt') }

START_MARKER  = '<S>'
END_MARKER    = '</S>'
UNKNOWN_TOKEN = '*UNKNOWN*'
UNKNOWN_LABEL = 'O'
#UNKNOWN_FEATURE = 'FEAT_NONE'

TEMP_DIR = join(ROOT_DIR, 'temp')

assert os.path.exists(SRL_CONLL_EVAL_SCRIPT)
if not os.path.exists(TEMP_DIR):
  os.makedirs(TEMP_DIR)

