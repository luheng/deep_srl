import random

from constants import *
from dictionary import Dictionary
import features

def get_sentences(filepath, use_se_marker=False):
  """ Read tokenized sentences from file """
  sentences = []
  with open(filepath) as f:
    for line in f.readlines():
      inputs = line.strip().split('|||')
      lefthand_input = inputs[0].strip().split()
      # If gold tags are not provided, create a sequence of dummy tags.
      righthand_input = inputs[1].strip().split() if len(inputs) > 1 \
                          else ['O' for _ in lefthand_input]
      if use_se_marker:
        words = [START_MARKER] + lefthand_input + [END_MARKER]
        labels = [None] + righthand_input + [None]
      else:
        words = lefthand_input
        labels = righthand_input
      sentences.append((words, labels))
  return sentences

    #lines = f.readlines()
    #sentences = [line.strip().split('|||') for line in lines]
    #if use_se_marker:
    #  return [([START_MARKER] + words.strip().split() + [END_MARKER], [None] + labels.strip().split() + [None])
    #      for words,labels in sentences]
    #else:
    #  return [(words.strip().split(), labels.strip().split()) for words,labels in sentences]
    
def get_srl_sentences(filepath, use_se_marker=False):
  """ Read tokenized SRL sentences from file.
    File format: {predicate_id} [word0, word1 ...] ||| [label0, label1 ...]
    Return:
      A list of sentences, with structure: [[words], predicate, [labels]]
  """
  sentences = []
  with open(filepath) as f:
    for line in f.readlines():
      inputs = line.strip().split('|||')
      lefthand_input = inputs[0].strip().split()
      # If gold tags are not provided, create a sequence of dummy tags.
      righthand_input = inputs[1].strip().split() if len(inputs) > 1 \
                          else ['O' for _ in lefthand_input[1:]]
      predicate = int(lefthand_input[0])
      if use_se_marker:
        words = [START_MARKER] + lefthand_input[1:] + [END_MARKER]
        labels = [None] + righthand_input + [None]
      else:
        words = lefthand_input[1:]
        labels = righthand_input
      sentences.append((words, predicate, labels))
  return sentences

def get_pretrained_embeddings(filepath):
  embeddings = dict()
  with open(filepath, 'r') as f:
    for line in f:
      info = line.strip().split()
      #lines = [line.strip().split() for line in f.readlines()]
      #embeddings = dict([(line[0], [float(r) for r in line[1:]]) for line in lines])
      embeddings[info[0]] = [float(r) for r in info[1:]]
    f.close()
  embedding_size = len(embeddings.values()[0])
  print 'Embedding size={}'.format(embedding_size)
  embeddings[START_MARKER] = [random.gauss(0, 0.01) for _ in range(embedding_size)]
  embeddings[END_MARKER] = [random.gauss(0, 0.01) for _ in range(embedding_size)]
  if not UNKNOWN_TOKEN in embeddings:
    embeddings[UNKNOWN_TOKEN] = [random.gauss(0, 0.01) for _ in range(embedding_size)]
  return embeddings

def string_sequence_to_ids(str_seq, dictionary, lowercase=False, pretrained_embeddings=None):
  """ If pretrained_embeddings is provided, strings not in the embeddings 
    Pretrained embeddings is a dictionary from strings to python list. 
  """
  ids = []
  for s in str_seq:
    if s is None:
      ids.append(-1)
      continue
    if lowercase:
      s = s.lower()
    if (pretrained_embeddings != None) and not (s in pretrained_embeddings) :
      s = UNKNOWN_TOKEN
    ids.append(dictionary.add(s))
  return ids
  
def get_postag_data(config, train_path, dev_path, vocab_path=None, label_path=None):
  use_se_marker = config.use_se_marker
  raw_train_sents = get_sentences(train_path, use_se_marker)
  raw_dev_sents = get_sentences(dev_path, use_se_marker)
  word_to_embeddings = get_pretrained_embeddings(WORD_EMBEDDINGS[config.word_embedding])
  
  # Prepare word dictionary.
  word_dict = Dictionary(unknown_token=UNKNOWN_TOKEN)
  if use_se_marker:
    word_dict.add_all([START_MARKER, END_MARKER])
  if vocab_path != None:
    with open(vocab_path, 'r') as f_vocab:
      for line in f_vocab:
        word_dict.add(line.strip())
      f_vocab.close()
    word_dict.accept_new = False
    print 'Load {} words. Dictionary freezed.'.format(word_dict.size())
  
  # Parpare label dictionary.
  label_dict = Dictionary() 
  if label_path != None:
    with open(label_path, 'r') as f_labels:
      for line in f_labels:
        label_dict.add(line.strip())
      f_labels.close()
    label_dict.set_unknown_token(UNKNOWN_LABEL)
    label_dict.accept_new = False
    print 'Load {} labels. Dictionary freezed.'.format(label_dict.size())

  train_sents = [(string_sequence_to_ids(sent[0], word_dict, True, word_to_embeddings),
          string_sequence_to_ids(sent[1], label_dict)) for sent in raw_train_sents]
  dev_sents = [(string_sequence_to_ids(sent[0], word_dict, True, word_to_embeddings),
          string_sequence_to_ids(sent[1], label_dict)) for sent in raw_dev_sents]

  print("Extracted {} words and {} tags".format(word_dict.size(), label_dict.size()))
  print("Max training sentence length: {}".format(max([len(s[0]) for s in train_sents])))
  print("Max development sentence length: {}".format(max([len(s[0]) for s in dev_sents])))
  word_embedding = [word_to_embeddings[w] for w in word_dict.idx2str]
  word_embedding_shape = [len(word_embedding), len(word_embedding[0])]
  return (train_sents, dev_sents, word_dict, label_dict, [word_embedding], [word_embedding_shape])

def get_srl_data(config, train_data_path, dev_data_path, vocab_path=None, label_path=None):
  '''
  '''
  use_se_marker = config.use_se_marker
  raw_train_sents = get_srl_sentences(train_data_path, use_se_marker)
  raw_dev_sents = get_srl_sentences(dev_data_path, use_se_marker)
  word_to_embeddings = get_pretrained_embeddings(WORD_EMBEDDINGS[config.word_embedding])

  # Prepare word dictionary.
  word_dict = Dictionary(unknown_token=UNKNOWN_TOKEN)
  if use_se_marker:
    word_dict.add_all([START_MARKER, END_MARKER])
  if vocab_path != None:
    with open(vocab_path, 'r') as f_vocab:
      for line in f_vocab:
        word_dict.add(line.strip())
      f_vocab.close()
    word_dict.accept_new = False
    print 'Load {} words. Dictionary freezed.'.format(word_dict.size())
  
  # Parpare label dictionary.
  label_dict = Dictionary() 
  if label_path != None:
    with open(label_path, 'r') as f_labels:
      for line in f_labels:
        label_dict.add(line.strip())
      f_labels.close()
    label_dict.set_unknown_token(UNKNOWN_LABEL)
    label_dict.accept_new = False
    print 'Load {} labels. Dictionary freezed.'.format(label_dict.size())

  # Get tokens and labels
  train_tokens = [string_sequence_to_ids(sent[0], word_dict, True, word_to_embeddings) for sent in raw_train_sents]
  train_labels = [string_sequence_to_ids(sent[2], label_dict) for sent in raw_train_sents]
  
  if label_dict.accept_new:
    label_dict.set_unknown_token(UNKNOWN_LABEL)
    label_dict.accept_new = False
  
  dev_tokens = [string_sequence_to_ids(sent[0], word_dict, True, word_to_embeddings) for sent in raw_dev_sents]
  dev_labels = [string_sequence_to_ids(sent[2], label_dict) for sent in raw_dev_sents]
  
  # Get features
  print 'Extracting features'
  train_features, feature_shapes = features.get_srl_features(raw_train_sents, config)
  dev_features, feature_shapes2 = features.get_srl_features(raw_dev_sents, config)
  for f1, f2 in zip(feature_shapes, feature_shapes2):
    assert f1 == f2
 
  # For additional features. Unused now. 
  feature_dicts = []
  for feature in config.features:
    feature_dicts.append(None)
  
  train_sents = []
  dev_sents = []
  for i in range(len(train_tokens)):
    train_sents.append((train_tokens[i],) + tuple(train_features[i]) + (train_labels[i],))
  for i in range(len(dev_tokens)):
    dev_sents.append((dev_tokens[i],) + tuple(dev_features[i]) + (dev_labels[i],))
  
  print("Extraced {} words and {} tags".format(word_dict.size(), label_dict.size()))
  print("Max training sentence length: {}".format(max([len(s[0]) for s in train_sents])))
  print("Max development sentence length: {}".format(max([len(s[0]) for s in dev_sents])))
  
  word_embedding = [word_to_embeddings[w] for w in word_dict.idx2str]
  word_embedding_shape = [len(word_embedding), len(word_embedding[0])]
  return (train_sents, dev_sents, word_dict, label_dict,
      [word_embedding, None, None],
      [word_embedding_shape] + feature_shapes,
      [word_dict, ] + feature_dicts)
  
def get_postag_test_data(filepath, config, word_dict, label_dict, allow_new_words=True):
  # New words are allowed as long as they are covered by pre-trained embeddings.
  word_dict.accept_new = allow_new_words
  if label_dict.accept_new:
    label_dict.set_unknown_token(UNKNOWN_LABEL)
    label_dict.accept_new = False
 
  if filepath != None and filepath != '': 
    samples = get_sentences(filepath, config.use_se_marker)
  else:
    samples = []
  word_to_embeddings = get_pretrained_embeddings(WORD_EMBEDDINGS[config.word_embedding])
  if allow_new_words:
    tokens = [string_sequence_to_ids(sent[0], word_dict, True, word_to_embeddings) for sent in samples]
  else:
    tokens = [string_sequence_to_ids(sent[0], word_dict, True) for sent in samples]
  labels = [string_sequence_to_ids(sent[1], label_dict) for sent in samples]
  sentences = []
  for i in range(len(tokens)):
    sentences.append((tokens[i],) + (labels[i],))
    
  word_embedding = [word_to_embeddings[w] for w in word_dict.idx2str]
  word_embedding_shape = [len(word_embedding), len(word_embedding[0])]
  return (sentences, [word_embedding], [word_embedding_shape])

def get_srl_test_data(filepath, config, word_dict, label_dict, allow_new_words=True):
  word_dict.accept_new = allow_new_words
  if label_dict.accept_new:
    label_dict.set_unknown_token(UNKNOWN_LABEL)
    label_dict.accept_new = False
  
  if filepath != None and filepath != '': 
    samples = get_srl_sentences(filepath, config.use_se_marker)
  else:
    samples = []
  word_to_embeddings = get_pretrained_embeddings(WORD_EMBEDDINGS[config.word_embedding])
  if allow_new_words:
    tokens = [string_sequence_to_ids(sent[0], word_dict, True, word_to_embeddings) for sent in samples]
  else:
    tokens = [string_sequence_to_ids(sent[0], word_dict, True) for sent in samples]
    
  labels = [string_sequence_to_ids(sent[2], label_dict) for sent in samples]
  srl_features, feature_shapes = features.get_srl_features(samples, config)
  
  sentences = []
  for i in range(len(tokens)):
    sentences.append((tokens[i],) + tuple(srl_features[i]) + (labels[i],))
    
  word_embedding = [word_to_embeddings[w] for w in word_dict.idx2str]
  word_embedding_shape = [len(word_embedding), len(word_embedding[0])]
  
  return (sentences,  [word_embedding, None, None], [word_embedding_shape,] + feature_shapes)
    
