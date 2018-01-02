# Deep Semantic Role Labeling

This repository contains code for training and using the deep SRL model described in:
[Deep Semantic Role Labeling: What works and what's next](https://homes.cs.washington.edu/~luheng/files/acl2017_hllz.pdf)

If you use our code, please cite our paper as follows:  
  > @inproceedings{he2017deep,  
  > &nbsp; &nbsp; title={Deep Semantic Role Labeling: What Works and What’s Next},  
  > &nbsp; &nbsp; author={He, Luheng and Lee, Kenton and Lewis, Mike and Zettlemoyer, Luke},  
  > &nbsp; &nbsp; booktitle={Proceedings of the Annual Meeting of the Association for Computational Linguistics},  
  > &nbsp; &nbsp; year={2017}  
  > }  

## Getting Started
### Prerequisites:
* python should be using Python 2. You can simulate this with virtualenv.
* pip install numpy
* pip install theano==0.9.0 (Compability with Theano 1.0 is not tested yet)
* pip install protobuf
* pip install nltk (For tokenization, required only for the interactive console)
* sudo apt-get install tcsh (Only required for processing CoNLL05 data)
* [Git Large File Storage] (https://git-lfs.github.com/): Required to download the large model files. Alternatively, you could get the models [here](https://drive.google.com/drive/folders/0B5zHXdvxrsjNZUx2YXJ5cEM0TW8?usp=sharing)
* [GloVe](https://nlp.stanford.edu/projects/glove/) embeddings and the [srlconll](http://www.lsi.upc.edu/~srlconll/soft.html) scripts:  
`./scripts/fetch_required_data.sh`

### Pretrained models
Decompress the models (in resources) under the neural_srl directory. For example, under the codebase directory:  
`tar -zxvf resources/conll05_model.tar.gz`

Here's a list of pretrained models:
* `conll05_model.tar.gz`: Single model trained on CoNLL-2005 dataset.
* `conll05_ensemble.tar.gz`: 5 model ensemble trained on CoNLL-2005 dataset.
* `conll05_propid_model.tar.gz`: Predicate identification model train on CoNLL-2005.
* `conll2012_model.tar.gz`: Single model trained on CoNLL-2012 dataset.
* `conll2012_ensemble.tar.gz`: 5 model ensemble trained on CoNLL-2012 dataset.
* `conll2012_propid_model.tar.gz`: Predicate identification model train on CoNLL-2012.


### Try out the interactive console!
`python python/interactive.py  --model conll05_model/ --pidmodel conll05_propid_model`

### End-to-end SRL prediction:
Run:  
`./scripts/run_end2end.sh sample_data/sentences_with_predicates.txt temp/sample.out` (on CPU)
or:   
`./scripts/run_end2end.sh sample_data/sentences_with_predicates.txt temp/sample.out ${gpu_id}` (on GPU)

Note that the script adds `/usr/local/cuda/...` to `PATH` and `CUDA_LD_LIBRARY_PATH`, and loads pretrained models from `./conll05_propid_model` and `./conll05_ensemble`, please adjust the configurations according to your own setup.

The input file contains tokenized sentences, one sentence per line.

The output file will contain something like:
> John told Pat to cut off the tree .  
>  Predicate: told(1)  
>    A0: John  
>    V: told  
>    A2: Pat  
>    A1: to cut off the tree  

> John told Pat to cut off the tree .  
>  Predicate: cut(4)  
>    A0: Pat  
>    V: cut off  
>    A1: the tree  


### Scalability Issue
* Building model for the first time might take a while (less then 30 minutes).
* Currently `predict.py` loads the entire input file into memory, so it would be better to keep the number of sentences in each file under 50,000.

## CoNLL Data
For replicating results on CoNLL-2005 and CoNLL-2012 datasets, please follow the steps below.

### CoNLL-2005
The data is provided by:
[CoNLL-2005 Shared Task](http://www.lsi.upc.edu/~srlconll/soft.html),
but the original words are from the Penn Treebank dataset, which is not publicly available.
If you have the PTB corpus, you can run:  
` ./scripts/fetch_and_make_conll05_data.sh  /path/to/ptb/`  

### CoNLL-2012
You have to follow the instructions below to get CoNLL-2012 data
[CoNLL-2012](http://cemantix.org/data/ontonotes.html), this would result in a directory called `/path/to/conll-formatted-ontonotes-5.0`.
Run:  
`./scripts/make_conll2012_data.sh /path/to/conll-formatted-ontonotes-5.0`

## Predicting SRL with trained model
See usage of `python/train.py`:  
`python python/predict.py -h`

Or as a quick start, run trained model (requires conll05_ensemble):  
`./scripts/run_predict_conll05.sh ${gpu_id}`
or:   
`./scripts/run_predict_conll05.sh` for running on CPU.

Run the model end-to-end with predicted (requires conll05_ensemble, and conll05_propid_model):  
`./scripts/run_end_to_end_conll05.sh ${gpu_id}`

Running the CoNLL-2012 model works similarly.

## Training a new model
See usage of `python/train.py`:  
`python python/train.py -h`

Train an SRL model (with gold predicates) with pre-defined config files:
`./scripts/run_train.sh ${gpu_id}`

Train a predicate identifider:
`./scripts/run_propid_train.sh ${gpu_id}`

Note that at training time, `train.py`runs in the `FAST_RUN` model, which will result in a huge overhead of model compilation. It might take up to several minutes for a 2 layer model, and up to 8 hours for an 8 layer model with variational dropout.

## Data Format
Please refer to the files in `sample_data` and the explanations below for how to format the model input. 

### BIO-tagging format for the SRL model
Each line contains exactly one training sample, which has the predicate information (index in the sentences, starting from 0), the tokenized sentence, and a sequence of tags. If gold tags do not exist, just use a sequence of Os. The sentence and the tag sequence is seperated with a ||| symbol. We use the [IOB2](https://en.wikipedia.org/wiki/Inside_Outside_Beginning) format. All the tokens and symbols are seperated by an arbitrary whitespace.

Example lines:
  > 2 My cats love hats . ||| B-A0 I-A0 B-V B-A1 O

### Tagging format for the predicate identication model
The format is similar to the above defined, except that each line corresponds to an input sentence, and no predicate information is provided. The prediates correspond to the V tags and all other words are labeled with O tags.

Example lines:
  > My cats love hats , they say . ||| O O V O O O V O

### Configuration for training.
`config` contains some configuration files for training the SRL model (`srl_config.json` and `srl_small_config.json`) as well as for training the predicate-id model (`propid_config.json`)


## Contact

Contact [Luheng He](https://homes.cs.washington.edu/~luheng/) if you have any questions!
