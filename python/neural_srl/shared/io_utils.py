from google.protobuf.internal import encoder

_EncodeVarint = encoder._VarintEncoder()

def write_delimited_to(out_file, message):
  msg_size = message.ByteSize()
  pieces = []
  _EncodeVarint(pieces.append, msg_size)
  out_file.write(b"".join(pieces)) 
  out_file.write(message.SerializeToString())

def read_gold_props(gold_props_file):
  """ Read gold predicates from CoNLL-formatted file.
  """
  gold_props = []
  props = []
  with open(gold_props_file, 'r') as f:
    for line in f:
      line = line.strip()
      if line == '':
        gold_props.append(props)
        props = []
      else:
        props.append(line.split()[0])
    f.close()
  if len(props) > 0:
    gold_props.append(props)
  return gold_props

def write_predprops_to(predictions, label_dict, input_file, output_file,
                       gold_props_file=None, output_props_file = None):
  """ Write predicted predicate information to files.

      Arguments:
        predictions: Predictions from the predicate identification model.
                      Is a numpy array of size [num_sentences, max_sentence_length].
        label_dict: Label dictionary.
        input_file: Input sequential tagging file.
        output_file: Output SRL file with identified predicates.
        gold_props_file: Input file with gold predicates in CoNLL format.
        output_props_file: Output SRL file with identified predicates, in CoNLL format.
  """

  fin = open(input_file, 'r')
  fout = open(output_file, 'w')

  if output_props_file != None and output_props_file != '':
    fout_props = open(output_props_file, 'w')
  else:
    fout_props = None

  if gold_props_file != None and gold_props_file != '':
    gold_props = read_gold_props(gold_props_file)
    print len(gold_props), len(predictions)
    assert len(gold_props) == len(predictions)
  else:
    gold_props = None

  sent_id = 0
  for line in fin:
    # Read original sentence from input file.
    raw_sent = line.split('|||')[0].strip()
    tokens = raw_sent.split(' ')
    slen = len(tokens)
    pred = predictions[sent_id, :slen]
    props = []

    for (t, p) in enumerate(pred):
      if label_dict.idx2str[p] == 'V':
        out_tags = ['O' for _  in range(slen)]
        out_tags[t] = 'B-V'
        out_line = str(t) + '\t' + raw_sent + ' ||| ' + ' '.join(out_tags) + '\n'
        fout.write(out_line)
        props.append(t)

    if fout_props != None:
      if sent_id > 0:
        fout_props.write('\n')
      for t in range(slen):
        lemma = 'P' + tokens[t].lower()
        # In order for CoNLL evaluation script to run, we need to output the same
        # lemma as the gold predicate in the CoNLL-formatted file. 
        if gold_props is not None and gold_props[sent_id][t] != '-':
          lemma = gold_props[sent_id][t]
        if t in props:
          fout_props.write(lemma)
        else:
          fout_props.write('-')
        for p in props:
          if t == p:
            fout_props.write('\t(V*)')
          else:
            fout_props.write('\t*')
        fout_props.write('\n')

    sent_id += 1

  fout.close()
  print ('Predicted predicates in sequential-tagging format written to: {}.'.format(output_file))
  if fout_props != None:
    fout_props.close()
    print ('CoNLL-formatted predicate information written to: {}.'.format(output_props_file))


def bio_to_spans(predictions, label_dict):
  """ Convert BIO-based predictions to a set of arguments.
      Arguments:
        predictions: A single integer array, already truncated to the original sequence lengths.
        label_dict: Label dictionary.
      Returns:
        A sequence of labeled arguments: [ ("ARG_LABEL", span_start, span_end), ... ], ordered by their positions.
  """
  args = []
  tags = [label_dict.idx2str[p] for p in predictions]
  for (i, tag) in enumerate(tags):
    if tag == 'O':
      continue
    label = tag[2:]
    # Append new span.
    if tag[0] == 'B' or len(args) == 0 or label != tags[i-1][2:]:
      args.append([label, i, -1])
    # Close current span.
    if i == len(predictions) - 1 or tags[i+1][0] == 'B' or label != tags[i+1][2:]:
      args[-1][2] = i
  return args


def print_to_readable(predictions, num_tokens, label_dict, input_path, output_path):
  """ Print predictions to human-readable format.
  """
  fout = open(output_path, 'w')
  sample_id = 0
  for line in open(input_path, 'r'):
    info = line.split('|||')[0].strip().split()
    pid = int(info[0])
    sent = info[1:]
    fout.write(' '.join(sent) + '\n')
    fout.write('\tPredicate: {}({})\n'.format(sent[pid], pid))

    tags = predictions[sample_id, :num_tokens[sample_id]]
    arg_spans = bio_to_spans(tags, label_dict)
    for arg in arg_spans:
      fout.write('\t\t{}: {}\n'.format(arg[0], " ".join(sent[arg[1]:arg[2]+1])))
    fout.write('\n')
    sample_id += 1 
    
  fout.close()
    
    


