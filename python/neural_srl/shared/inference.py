import numpy

def get_transition_params(label_strs):
  '''Construct transtion scoresd (0 for allowed, -inf for invalid).
  Args:
    label_strs: A [num_tags,] sequence of BIO-tags.
  Returns:
    A [num_tags, num_tags] matrix of transition scores.  
  '''
  num_tags = len(label_strs)
  transition_params = numpy.zeros([num_tags, num_tags], dtype=numpy.float32)
  for i, prev_label in enumerate(label_strs):
    for j, label in enumerate(label_strs):
      if i != j and label[0] == 'I' and not prev_label == 'B' + label[1:]:
        transition_params[i,j] = numpy.NINF
  return transition_params

def viterbi_decode(score, transition_params):
  """ Adapted from Tensorflow implementation.
  Decode the highest scoring sequence of tags outside of TensorFlow.
  This should only be used at test time.
  Args:
    score: A [seq_len, num_tags] matrix of unary potentials.
    transition_params: A [num_tags, num_tags] matrix of binary potentials.
  Returns:
    viterbi: A [seq_len] list of integers containing the highest scoring tag
        indicies.
    viterbi_score: A float containing the score for the Viterbi sequence.
  """
  trellis = numpy.zeros_like(score)
  backpointers = numpy.zeros_like(score, dtype=numpy.int32)
  trellis[0] = score[0]
  for t in range(1, score.shape[0]):
    v = numpy.expand_dims(trellis[t - 1], 1) + transition_params
    trellis[t] = score[t] + numpy.max(v, 0)
    backpointers[t] = numpy.argmax(v, 0)
  viterbi = [numpy.argmax(trellis[-1])]
  for bp in reversed(backpointers[1:]):
    viterbi.append(bp[viterbi[-1]])
  viterbi.reverse()
  viterbi_score = numpy.max(trellis[-1])
  return viterbi, viterbi_score

