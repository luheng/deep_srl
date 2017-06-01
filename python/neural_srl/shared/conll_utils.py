def bio_to_se(labels):
  slen = len(labels)
  new_labels = []
  has_opening = False
  for i in range(slen):
    label = labels[i]
    if label == 'O':
      new_labels.append('*')
      continue
    new_label = '*'
    if label[0] == 'B' or i == 0 or label[1:] != labels[i-1][1:]:
      new_label = '(' + label[2:] + new_label
      has_opening = True
    if i == slen - 1 or labels[i+1][0] == 'B' or label[1:] != labels[i+1][1:]:
      new_label = new_label + ')'
      has_opening = False
    new_labels.append(new_label)
  
  if has_opening:
    ''' logging '''
    print("Has unclosed opening: {}".format(labels))
  return new_labels

def print_sentence_to_conll(fout, tokens, labels):
  for label_column in labels:
    assert len(label_column) == len(tokens)
  for i in range(len(tokens)):
    fout.write(tokens[i].ljust(15))
    for label_column in labels:
      fout.write(label_column[i].rjust(15))
    fout.write("\n")
  fout.write("\n")
  
def print_to_conll(pred_labels, gold_props_file, output_filename):
  """ 
  """
  fout = open(output_filename, 'w')
  seq_ptr = 0
  num_props_for_sentence = 0
  tokens_buf = []
  
  for line in open(gold_props_file, 'r'):
    line = line.strip()
    if line == "" and len(tokens_buf) > 0:
      print_sentence_to_conll(fout, tokens_buf, pred_labels[seq_ptr:seq_ptr+num_props_for_sentence])
      seq_ptr += num_props_for_sentence
      tokens_buf = []
      num_props_for_sentence = 0
    else:
      info = line.split()
      num_props_for_sentence = len(info) - 1
      tokens_buf.append(info[0])
      
  # Output last sentence. 
  if len(tokens_buf) > 0:
    print_sentence_to_conll(fout, tokens_buf, pred_labels[seq_ptr:seq_ptr+num_props_for_sentence])
    
  fout.close()

def print_gold_to_conll(data, word_dict, label_dict, output_filename):
  fout = open(output_filename, 'w')
  props_buf = []
  labels_buf = []
  tokens_buf = []
  prev_words = ''
  
  x, y, num_tokens, _ = data
  for (sent, gold, slen) in zip(x, y, num_tokens):
    words = [word_dict.idx2str[w[0]] for w in sent[:slen]]
    labels = [label_dict.idx2str[l] for l in gold[:slen]]
    
    concat_words = ' '.join(words)
    if concat_words != prev_words and len(props_buf) > 0:
      tokens = [w if i in props_buf else '-' for i,w in enumerate(tokens_buf)]
      
      print_sentence_to_conll(fout, tokens, labels_buf)
      props_buf = []
      tokens_buf = []
      labels_buf = []
      prev_words = ''
    
    if prev_words == '':
      prev_words = concat_words
      tokens_buf = [w for w in words]
    if 'B-V' in labels:
      prop_id = labels.index('B-V')
      props_buf.append(prop_id)
      labels_buf.append(bio_to_se(labels))

  if len(props_buf) > 0:
    tokens = [w if i in props_buf else '-' for i,w in enumerate(tokens_buf)]
    print_sentence_to_conll(fout, tokens, labels_buf) 
        
  fout.close()
