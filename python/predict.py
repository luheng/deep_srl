''' Predict and output scores.

   - Reads model param file.
   - Runs data.
   - Remaps label indices.
   - Outputs protobuf file.
'''

from neural_srl.shared import *
from neural_srl.shared.constants import *
from neural_srl.shared.conll_utils import print_to_conll
from neural_srl.shared.dictionary import Dictionary
from neural_srl.shared.inference import *
from neural_srl.shared.io_utils import *
from neural_srl.shared.tagger_data import TaggerData
from neural_srl.shared.measurements import Timer
from neural_srl.shared.evaluation import PropIdEvaluator, SRLEvaluator
from neural_srl.shared.tensor_pb2 import *
from neural_srl.shared.scores_pb2 import *
from neural_srl.theano.tagger import BiLSTMTaggerModel
from neural_srl.theano.util import floatX

import argparse
import numpy
import os
import sys
import theano

def get_scores(config, task, model_path, word_dict_path, label_dict_path, input_path):  
  with Timer('Data loading'):
    print ('Task: {}'.format(task))
    allow_new_words = True
    print ('Allow new words in test data: {}'.format(allow_new_words))
  
    # Load word and tag dictionary
    word_dict = Dictionary(unknown_token=UNKNOWN_TOKEN)
    label_dict = Dictionary()
    word_dict.load(word_dict_path)
    label_dict.load(label_dict_path)
    data = TaggerData(config, [], [], word_dict, label_dict, None, None)

    # Load test data.
    if task == 'srl':
      test_sentences, emb_inits, emb_shapes = reader.get_srl_test_data(
                                                    input_path,
                                                    config,
                                                    data.word_dict,
                                                    data.label_dict,
                                                    allow_new_words)
    else:
      test_sentences, emb_inits, emb_shapes = reader.get_postag_test_data(
                                                    input_path,
                                                    config,
                                                    data.word_dict,
                                                    data.label_dict,
                                                    allow_new_words)
    
    print ('Read {} sentences.'.format(len(test_sentences)))
  
    # Add pre-trained embeddings for new words in the test data.
    #if allow_new_words:
    data.embedding_shapes = emb_shapes
    data.embeddings = emb_inits

    # Batching.
    test_data = data.get_test_data(test_sentences, batch_size=config.dev_batch_size)
      
  with Timer('Model building and loading'):
    model = BiLSTMTaggerModel(data, config=config, fast_predict=True)
    model.load(model_path)
    dist_function = model.get_distribution_function()
     
  with Timer('Running model'):
    scores = None
    for i, batched_tensor in enumerate(test_data):
      x, _, num_tokens, weights = batched_tensor
      p, sc = dist_function(x, weights)
      scores = numpy.concatenate((scores, sc), axis=0) if i > 0 else sc
   
  return scores, data, test_sentences, test_data

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--model',
                      type=str,
                      default='',
                      required=True, 
                      help='Path to the model directory.')

 
  parser.add_argument('--input',
                      type=str,
                      default='',
                      required=True, 
                      help='Path to the input file path (sequetial tagging format).')

  parser.add_argument('--task',
                       type=str,
                       help='Training task (srl or propid). Default is srl.',
                       default='srl',
                       choices=['srl', 'propid'])

  parser.add_argument('--gold',
                      type=str,
                      default='',
                      help='(Optional) Path to the file containing gold propositions (provided by CoNLL shared task).')

  parser.add_argument('--inputprops',
                      type=str,
                      default='',
                      help='(Optional) Path to the predicted predicates in CoNLL format. Ignore if using gold predicates.')

  parser.add_argument('--output',
                      type=str,
                      default='',
                      help='(Optional) Path for output predictions.')

  parser.add_argument('--outputprops',
                      type=str,
                      default='',
                      help='(Optional) Path for output predictions in CoNLL format. Only used when task is {propid}.')

  parser.add_argument('--proto',
                      type=str,
                      default='',
                      help='(Optional) Path to the proto file path (for reusing predicted scores).')


  args = parser.parse_args()
  config = configuration.get_config(os.path.join(args.model, 'config'))

  # Detect available ensemble models.
  num_ensemble_models = 1
  for i in range(20):
    model_path = os.path.join(args.model, 'model{}.npz'.format(i))
    if os.path.exists(model_path):
      num_ensemble_models = i + 1
    else:
      break
  if num_ensemble_models == 1:
    print ('Using single model.')
  else:
    print ('Using an ensemble of {} models'.format(num_ensemble_models))

  ensemble_scores = None
  for i in range(num_ensemble_models):
    if num_ensemble_models == 1:
      model_path = os.path.join(args.model, 'model.npz')
      word_dict_path = os.path.join(args.model, 'word_dict')
    else:
      model_path = os.path.join(args.model, 'model{}.npz'.format(i))
      word_dict_path = os.path.join(args.model, 'word_dict{}'.format(i))
    label_dict_path = os.path.join(args.model, 'label_dict')

    # Compute local scores.
    scores, data, test_sentences, test_data = get_scores(config,
                                                         args.task,
                                                         model_path,
                                                         word_dict_path,
                                                         label_dict_path,
                                                         args.input)
    ensemble_scores = numpy.add(ensemble_scores, scores) if i > 0 else scores

  # Getting evaluator
  gold_props_file = args.gold if args.gold != '' else None
  pred_props_file = args.inputprops if args.inputprops != '' else None

  if args.task == 'srl':
    evaluator = SRLEvaluator(data.get_test_data(test_sentences, batch_size=None),
                             data.label_dict,
                             gold_props_file,
                             use_se_marker=config.use_se_marker,
                             pred_props_file=pred_props_file,
                             word_dict=data.word_dict)
  else:
    evaluator = PropIdEvaluator(data.get_test_data(test_sentences, batch_size=None),
                                data.label_dict) 

  if args.proto != '':
    print 'Writing to proto {}'.format(args.proto)
    pb_file = open(args.proto, 'wb')      
  else:
    pb_file = None

  with Timer("Decoding"):
    transition_params = get_transition_params(data.label_dict.idx2str)
    num_tokens = None

    # Collect sentence length information
    for (i, batched_tensor) in enumerate(test_data):
      _, _, nt, _ = batched_tensor
      num_tokens = numpy.concatenate((num_tokens, nt), axis=0) if i > 0 else nt

    # Decode.
    if num_ensemble_models > 1:
      ensemble_scores = numpy.divide(ensemble_scores, 1.0 * num_ensemble_models)

    predictions = []
    line_counter = 0
    for i, slen in enumerate(num_tokens):
      sc = ensemble_scores[i, :slen, :]

      if args.task == 'srl':
        pred, _ = viterbi_decode(sc, transition_params)
      else:
        pred = numpy.argmax(sc, axis=1)

      batch_pred = numpy.array(pred)
      batch_pred.resize(ensemble_scores.shape[1])
      predictions.append(batch_pred)

      # Construct protobuf message
      if pb_file != None:
        sample_id = line_counter
        sent_sc = SentenceScoresProto(sentence_id=sample_id,
                                      scores=TensorProto(dimensions=DimensionsProto(
                                          dimension=sentence_scores.shape),
                                      value=sentence_scores.flatten()))
        write_delimited_to(pb_file, sent_sc)
      line_counter += 1

    if pb_file != None:
      pb_file.close()

  # Evaluate
  predictions = numpy.stack(predictions, axis=0)
  evaluator.evaluate(predictions)

  if args.task == 'srl' and args.output != '':
    print ('Writing to human-readable file: {}'.format(args.output))
    _, _, nt, _ = evaluator.data 
    print_to_readable(predictions, nt, data.label_dict, args.input, args.output)

  if args.task == 'propid':
    write_predprops_to(predictions, data.label_dict, args.input, args.output, args.gold,
                       args.outputprops)
   

