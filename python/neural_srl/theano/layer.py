import theano
import theano.tensor as tensor
from theano.ifelse import ifelse
from theano.sandbox.rng_mrg import MRG_RandomStreams

from ..shared.numpy_utils import *
from ..shared.constants import RANDOM_SEED
from util import *

def get_rnn_layer(layer_name):
  """ Predefined names for different LSTM variations.
  """
  if layer_name in {'highway', 'eu', 'hwlstm'}:
    return HighwayLSTMLayer
  if layer_name in {'simple', 'simplehw'}:
    return SimpleHighwayLSTMLayer
  if layer_name == 'residual':
    return ResidualLSTMLayer
  if layer_name == 'ran':
    return RANLayer
  if layer_name == 'hwran':
    return HighwayRANLayer
  return LSTMLayer


def _slice(_x, n, dim):
  if _x.ndim == 3:
    return _x[:, :, n*dim : (n+1)*dim]
  return _x[:, n*dim : (n+1)*dim]

def _p(pp, name):
  return '%s_%s' % (pp, name)

class EmbeddingLayer(object):
  """ Embedding layer with concatenated features.
  """
  def __init__(self, embedding_shapes, embedding_inits=None, prefix='embedding'):
    self.embedding_shapes = embedding_shapes
    self.num_feature_types = len(embedding_shapes)
    self.output_size = sum([shape[1] for shape in self.embedding_shapes])
    print("Using {} feature types, projected output dim={}.".format(self.num_feature_types,
                                                                    self.output_size))
    self.embeddings = [get_variable(_p(prefix, i), shape, random_normal_initializer(0.0, 0.01))
                         for i, shape in enumerate(embedding_shapes)]
    # Initialize embeddings with pretrained values
    if embedding_inits != None:
      for emb, emb_init in zip(self.embeddings, embedding_inits):
        if emb_init != None:
          emb.set_value(numpy.array(emb_init, dtype=floatX))
    self.params = self.embeddings
  
  def connect(self, inputs):
    features = [None] * self.num_feature_types
    for i in range(self.num_feature_types):
      indices = inputs[:,:,i].flatten()
      proj_shape = [inputs.shape[0], inputs.shape[1], self.embedding_shapes[i][1]]
      features[i] = self.embeddings[i][indices].reshape(proj_shape)

    if self.num_feature_types == 1:
      return features[0]
    return tensor.concatenate(features, axis=2)
  
class SoftmaxLayer(object):
  def __init__(self, input_dim, label_space_size, prefix='softmax'):
    self.input_dim = input_dim
    self.label_space_size = label_space_size
    self.W = get_variable(_p(prefix, 'W'), [input_dim, self.label_space_size],
                          random_normal_initializer(0.0, 0.01))
    self.b = get_variable(_p(prefix, 'b'), [self.label_space_size],
                          all_zero_initializer())
    self.params = [self.W, self.b]
    
  def connect(self, inputs):
    energy = tensor.dot(inputs, self.W) + self.b
    energy = energy.reshape([energy.shape[0] * energy.shape[1], energy.shape[2]])
    log_scores = tensor.log(tensor.nnet.softmax(energy))
    predictions = tensor.argmax(log_scores, axis=-1)
    return (log_scores, predictions)
    
class CrossEntropyLoss(object):
  def connect(self, inputs, weights, labels):
    """ - inputs: flattened log scores from the softmax layer.
    """    
    y_flat = labels.flatten()
    x_flat_idx = tensor.arange(y_flat.shape[0])
    cross_ent = - inputs[x_flat_idx, y_flat].reshape([labels.shape[0], labels.shape[1]])
    if weights != None:
      cross_ent = cross_ent * weights
    # Summed over timesteps. Averaged across samples in the batch.
    return cross_ent.sum(axis=0).mean() 
    
class LSTMLayer(object):
  """ Basic LSTM. From the LSTM Tutorial.
  """
  def __init__(self, input_dim, hidden_dim, forget_bias = 1.0,
               input_dropout_prob = 0.0, recurrent_dropout_prob = 0.0,
               use_orthnormal_init = True,
               fast_predict=False,
               prefix='lstm'):
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.forget_bias = forget_bias
    self.prefix = prefix
    self.fast_predict = fast_predict
    self._init_parameters(4, 4, use_orthnormal_init) 
    self._init_dropout_layers(input_dropout_prob, recurrent_dropout_prob)

  def _init_parameters(self, num_i2h, num_h2h, use_orthnormal_init):
    """ num_i2h: Number of input-to-hidden projection blocks.
        num_h2h: Number of hidden-to-hidden projection blocks.
    """
    input_dim = self.input_dim
    hidden_dim = self.hidden_dim
    if use_orthnormal_init:
      self.W = get_variable(_p(self.prefix, 'W'), [input_dim, num_i2h * hidden_dim],
                            block_orth_normal_initializer([input_dim,], [hidden_dim] * num_i2h))
      self.U = get_variable(_p(self.prefix, 'U'), [hidden_dim, num_h2h * hidden_dim],
                            block_orth_normal_initializer([hidden_dim,], [hidden_dim] * num_h2h))
    else:
      self.W = get_variable(_p(self.prefix, 'W'), [input_dim, num_i2h * hidden_dim],
                            random_normal_initializer(0.0, 0.01))
      self.U = get_variable(_p(self.prefix, 'U'), [hidden_dim, num_h2h * hidden_dim],
                            random_normal_initializer(0.0, 0.01))
    self.b = get_variable(_p(self.prefix, 'b'), [num_i2h * hidden_dim], all_zero_initializer())
    self.params = [self.W, self.U, self.b]
    
  def _init_dropout_layers(self, input_dropout_prob, recurrent_dropout_prob):
    self.input_dropout_layer = None
    self.recurrent_dropout_layer = None
    # TODO: Input dropout layer?

    """ Variational dropout for LSTM.
        Reference: A Theoretically Grounded Application of Dropout in Recurrent Neural Networks
            Gal and Ghahramani, 2016.
            https://arxiv.org/abs/1512.05287
    """
    if recurrent_dropout_prob > 0:
      self.recurrent_dropout_layer = DropoutLayer(recurrent_dropout_prob,
                                                  fix_mask=True,
                                                  fast_predict=self.fast_predict,
                                                  prefix='{}_rdrop'.format(self.prefix))
  
  def _step(self, x_, m_, h_, c_):
    preact = tensor.dot(h_, self.U) + x_

    i = tensor.nnet.sigmoid(_slice(preact, 0, self.hidden_dim))
    f = tensor.nnet.sigmoid(_slice(preact, 1, self.hidden_dim) + self.forget_bias)
    o = tensor.nnet.sigmoid(_slice(preact, 2, self.hidden_dim))
    j = tensor.tanh(_slice(preact, 3, self.hidden_dim))

    c = f * c_ + i * j
    c = m_[:, None] * c + (1. - m_)[:, None] * c_

    h = o * tensor.tanh(c)
    if self.recurrent_dropout_layer != None:
      h = self.recurrent_dropout_layer.connect(h, self.is_train)
    h = m_[:, None] * h + (1. - m_)[:, None] * h_

    return h, c
        
  def connect(self, inputs, mask, is_train):
    """ is_train: A boolean tensor.
    """
    max_length = inputs.shape[0]
    batch_size = inputs.shape[1]
    outputs_info = [tensor.alloc(numpy_floatX(0.), batch_size, self.hidden_dim),
            tensor.alloc(numpy_floatX(0.), batch_size, self.hidden_dim)]
    # Dropout mask sharing for variational dropout.
    self.is_train = is_train
    if self.recurrent_dropout_layer != None:
      self.recurrent_dropout_layer.generate_mask([batch_size, self.hidden_dim], is_train)
    
    inputs = tensor.dot(inputs, self.W) + self.b
    rval, _ = theano.scan(self._step, # Scan function
                sequences=[inputs, mask], # Input sequence
                outputs_info=outputs_info,
                name=_p(self.prefix, '_layers'),
                n_steps=max_length) # scan steps
    return rval[0]

class HighwayLSTMLayer(LSTMLayer):
  """ Highway LSTM. Reference: Training Very Deep Networks.
         Srivastava et al., 2015
         https://arxiv.org/abs/1507.06228
  """
  def __init__(self, input_dim, hidden_dim, forget_bias = 1.0,
               input_dropout_prob = 0.0, recurrent_dropout_prob = 0.0,
               use_orthnormal_init = True,         
               fast_predict=False,
               prefix='hwlstm'):
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.forget_bias = forget_bias
    self.fast_predict = fast_predict
    self.prefix = prefix
    self._init_parameters(6, 5, use_orthnormal_init)
    self._init_dropout_layers(input_dropout_prob, recurrent_dropout_prob)
  
  def _step(self, x_, m_, h_, c_):
    preact= tensor.dot(h_, self.U) + _slice(x_, 0, self.hidden_dim * 5)
    # i: input. f: forget. o: output. t: transform.
    # j: input w\ non-linearity. k: input w\o non-linearity.
    i = tensor.nnet.sigmoid(_slice(preact, 0, self.hidden_dim))
    f = tensor.nnet.sigmoid(_slice(preact, 1, self.hidden_dim) + self.forget_bias)
    o = tensor.nnet.sigmoid(_slice(preact, 2, self.hidden_dim))
    t = tensor.nnet.sigmoid(_slice(preact, 3, self.hidden_dim))
    j = tensor.tanh(_slice(preact, 4, self.hidden_dim))
    k = _slice(x_, 5, self.hidden_dim)

    c = f * c_ + i * j
    c = m_[:, None] * c + (1. - m_)[:, None] * c_

    h = t * o * tensor.tanh(c) + (1. - t) * k
    if self.recurrent_dropout_layer != None:
      h = self.recurrent_dropout_layer.connect(h, self.is_train)
    h = m_[:, None] * h + (1. - m_)[:, None] * h_
    
    return h, c
  
  def connect(self, inputs, mask, is_train):
    return LSTMLayer.connect(self, inputs, mask, is_train)

class HighwayRANLayer(LSTMLayer):
  """ Highway RAN layer.
  """
  def __init__(self, input_dim, hidden_dim, forget_bias = 1.0,
               input_dropout_prob = 0,
               recurrent_dropout_prob = 0,
               use_orthnormal_init = True,
               fast_predict=False,
               prefix='hwran'):
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.forget_bias = forget_bias
    self.prefix = prefix
    self.fast_predict = fast_predict
    self._init_parameters(4, 3, use_orthnormal_init)
    self._init_dropout_layers(input_dropout_prob, recurrent_dropout_prob)
  
  def _step(self, x_, m_, h_, c_):
    preact = tensor.dot(h_, self.U) + _slice(x_, 0, self.hidden_dim * 3)
    i = tensor.nnet.sigmoid(_slice(preact, 0, self.hidden_dim))
    f = tensor.nnet.sigmoid(_slice(preact, 1, self.hidden_dim) + self.forget_bias)
    t = tensor.nnet.sigmoid(_slice(preact, 2, self.hidden_dim))
    j = _slice(x_, 3, self.hidden_dim)

    c = i * j + f * c_
    c = m_[:, None] * c + (1. - m_)[:, None] * c_
    h = t * tensor.tanh(c) + (1. - t) * j
    if self.recurrent_dropout_layer != None:
      h = self.recurrent_dropout_layer.connect(h, self.is_train) 
    h = m_[:, None] * h + (1. - m_)[:, None] * h_

    return h, c
  
  def connect(self, inputs, mask, is_train):
    return LSTMLayer.connect(self, inputs, mask, is_train)

class SimpleHighwayLSTMLayer(LSTMLayer):
  """ Highway LSTM minus a linear projection.
  """
  def __init__(self, input_dim, hidden_dim, forget_bias = 1.0,
               input_dropout_prob=0.0,
               recurrent_dropout_prob=0.0,
               use_orthnormal_init=True,
               fast_predict=False,
               prefix='simplehw'):
    if input_dim != hidden_dim:
      raise NotImplementedError("Input dimension needs to be same as hidden!")
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.forget_bias = forget_bias
    self.prefix = prefix
    self.fast_predict = fast_predict
    self._init_parameters(5, 5, use_orthnormal_init)
    self._init_dropout_layers(input_dropout_prob, recurrent_dropout_prob)
  
  def _step(self, x_, px_, m_, h_, c_):
    preact = tensor.dot(h_, self.U) + px_
    # i: input. f: forget. o: output. t: transform.
    # j: input w\ non-linearity. k: input w\o non-linearity.
    i = tensor.nnet.sigmoid(_slice(preact, 0, self.hidden_dim))
    f = tensor.nnet.sigmoid(_slice(preact, 1, self.hidden_dim) + self.forget_bias)
    o = tensor.nnet.sigmoid(_slice(preact, 2, self.hidden_dim))
    t = tensor.nnet.sigmoid(_slice(preact, 3, self.hidden_dim))
    j = tensor.tanh(_slice(preact, 4, self.hidden_dim))
    
    c = f * c_ + i * j
    c = m_[:, None] * c + (1. - m_)[:, None] * c_

    h = t * o * tensor.tanh(c) + (1. - t) * x_
    if self.recurrent_dropout_layer != None:
      h = self.recurrent_dropout_layer.connect(h, self.is_train) 
    h = m_[:, None] * h + (1. - m_)[:, None] * h_
    
    return h, c
  
  def connect(self, inputs, mask, is_train):
    max_length = inputs.shape[0]
    batch_size = inputs.shape[1]
    outputs_info = [tensor.alloc(numpy_floatX(0.), batch_size, self.hidden_dim),
            tensor.alloc(numpy_floatX(0.), batch_size, self.hidden_dim)]

    # Dropout layers
    self.is_train = is_train
    if self.recurrent_dropout_layer != None:
      self.recurrent_dropout_layer.generate_mask([batch_size, self.hidden_dim], is_train)
    
    proj_inputs = tensor.dot(inputs, self.W) + self.b
    rval, _ = theano.scan(self._step, # Scan function
                sequences=[inputs, proj_inputs, mask], # Input sequence
                outputs_info=outputs_info,
                name=_p(self.prefix, '_layers'),
                n_steps=max_length) # scan steps
    return rval[0]

class ResidualLSTMLayer(SimpleHighwayLSTMLayer):
  """ Residual network as described in the Google MT paper,
      Wu et al. 2016.
  """
  def __init__(self, input_dim, hidden_dim, forget_bias = 1.0,
               input_dropout_prob=0.0,
               recurrent_dropout_prob=0.0,
               use_orthnormal_init=True,
               fast_predict=False,
               prefix='residual'):
    if input_dim != hidden_dim:
      raise NotImplementedError("Input dimension needs to be same as hidden!")
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.forget_bias = forget_bias
    self.prefix = prefix
    self.fast_predict = fast_predict
    self._init_parameters(4, 4, use_orthnormal_init)
    self._init_dropout_layers(input_dropout_prob, recurrent_dropout_prob)
  
  def _step(self, x_, px_, m_, h_, c_):  
    preact = tensor.dot(h_, self.U) + px_
    i = tensor.nnet.sigmoid(_slice(preact, 0, self.hidden_dim))
    f = tensor.nnet.sigmoid(_slice(preact, 1, self.hidden_dim) + self.forget_bias)
    o = tensor.nnet.sigmoid(_slice(preact, 2, self.hidden_dim))
    j = tensor.tanh(_slice(preact, 3, self.hidden_dim))
    
    c = f * c_ + i * j
    c = m_[:, None] * c + (1. - m_)[:, None] * c_
    
    # Residual connection.
    h = o * tensor.tanh(c) + x_
    if self.recurrent_dropout_layer != None:
      h = self.recurrent_dropout_layer.connect(h, self.is_train)
    h = m_[:, None] * h + (1. - m_)[:, None] * h_
    return h, c
  
  def connect(self, inputs, mask, is_train):
    return SimpleHighwayLSTMLayer.connect(self, inputs, mask, is_train)
    
class RANLayer(LSTMLayer):
  """ Recurrent Addivitve Networks (RAN), Lee et al., 2017
  """
  def __init__(self, input_dim, hidden_dim, forget_bias = 1.0,
               input_dropout_prob = 0,
               recurrent_dropout_prob = 0,
               use_orthnormal_init=True,
               fast_predict=False,
               prefix='ran'):
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.forget_bias = forget_bias
    self.prefix = prefix
    self.fast_predict = fast_predict
    self._init_parameters(3, 2, use_orthnormal_init)
    self._init_dropout_layers(input_dropout_prob, recurrent_dropout_prob)
  
  def _step(self, x_, m_, h_, c_):
    preact = tensor.dot(h_, self.U) + _slice(x_, 0, self.hidden_dim * 2)
    i = tensor.nnet.sigmoid(_slice(preact, 0, self.hidden_dim))
    f = tensor.nnet.sigmoid(_slice(preact, 1, self.hidden_dim) + self.forget_bias)
    # Linear-projected input.
    j = _slice(x_, 2, self.hidden_dim)
    c = i * j + f * c_
    c = m_[:, None] * c + (1. - m_)[:, None] * c_
    h = tensor.tanh(c)
    if self.recurrent_dropout_layer != None:
      h = self.recurrent_dropout_layer.connect(h, self.is_train) 
    return h, c
  
  def connect(self, inputs, mask, is_train):
    return LSTMLayer.connect(self, inputs, mask, is_train)

class DropoutLayer(object):
  def __init__(self, dropout_prob, fix_mask=False, fast_predict=False, prefix="dropout"):
    self.dropout_prob = dropout_prob
    self.fix_mask = fix_mask
    self.prefix = prefix
    self.fast_predict = fast_predict
    print (self.prefix, self.dropout_prob, self.fix_mask)
    assert (dropout_prob > 0)
    """ This one works for the scan function.
        (instead of theano.tensor.shared.randomstreams.RandomStreams)
        See discussion: https://groups.google.com/forum/#!topic/theano-users/DbvTgTqkT8o
    """
    self.rng = MRG_RandomStreams(seed=RANDOM_SEED, use_cuda=True)
    
  def generate_mask(self, mask_shape, is_train):
    if not self.fast_predict:
      self.dropout_mask = self.rng.binomial(n=1, p=1-self.dropout_prob,
                                            size=tuple(mask_shape),
                                            dtype=floatX)
    
  def connect(self, inputs, is_train):
    """ Trick to speed up model compiling at decoding time.
        (Avoids building a complicated CG.)
    """
    if not self.fix_mask:
      self.generate_mask(inputs.shape, is_train)
     
    if self.fast_predict:
      return inputs * (1 - self.dropout_prob)

    return ifelse(is_train,
            inputs * self.dropout_mask,
            inputs * (1 - self.dropout_prob))
  
