from optimizer import *
from layer import *

from collections import OrderedDict
import itertools
import numpy
import theano
import theano.tensor as tensor

class BiLSTMTaggerModel(object):
  """ Constructs the network and builds the following Theano functions:
      - pred_function: Takes input and mask, returns prediction.
      - loss_function: Takes input, mask and labels, returns the final cross entropy loss (scalar).
  """
  def __init__(self, data, config, fast_predict=False):
    self.embedding_shapes = data.embedding_shapes;
    self.lstm_type = config.lstm_cell  
    self.lstm_hidden_size = int(config.lstm_hidden_size)
    self.num_lstm_layers = int(config.num_lstm_layers)
    self.max_grad_norm = float(config.max_grad_norm)

    self.vocab_size = data.word_dict.size()
    self.label_space_size = data.label_dict.size()
    self.unk_id = data.unk_id
    
    # Initialize layers and parameters
    self.embedding_layer = EmbeddingLayer(data.embedding_shapes, data.embeddings)    
    self.params = [p for p in self.embedding_layer.params]
    
    self.rnn_layers = [None] * self.num_lstm_layers
    for l in range(self.num_lstm_layers):
      input_dim = self.embedding_layer.output_size if l == 0 else self.lstm_hidden_size
      input_dropout = config.input_dropout_prob if (config.per_layer_dropout or l == 0) else 0.0
      recurrent_dropout = config.recurrent_dropout_prob
      
      self.rnn_layers[l] = get_rnn_layer(self.lstm_type)(input_dim,
                                 self.lstm_hidden_size,
                                 input_dropout_prob=input_dropout,
                                 recurrent_dropout_prob=recurrent_dropout,
                                 fast_predict=fast_predict,
                                 prefix='lstm_{}'.format(l))
      print (self.rnn_layers[l])
      self.params.extend(self.rnn_layers[l].params)
    
    self.softmax_layer = SoftmaxLayer(self.lstm_hidden_size, self.label_space_size)
    self.params.extend(self.softmax_layer.params)
    
    # Build model
    # Shape of x: [seq_len, batch_size, num_features]
    self.x0 = tensor.ltensor3('x')
    self.y0 = tensor.lmatrix('y')
    self.mask0 = tensor.matrix('mask', dtype=floatX)
    self.is_train = tensor.bscalar('is_train')
    
    self.x = self.x0.dimshuffle(1, 0, 2)
    self.y = self.y0.dimshuffle(1, 0)
    self.mask = self.mask0.dimshuffle(1, 0) 
    
    self.inputs = [None] * (self.num_lstm_layers + 1)
    self.inputs[0] = self.embedding_layer.connect(self.x)
    self.rev_mask = self.mask[::-1]
    
    for l, rnn in enumerate(self.rnn_layers):
      outputs = rnn.connect(self.inputs[l],
                  self.mask if l % 2 == 0 else self.rev_mask,
                  self.is_train)
      self.inputs[l+1] = outputs[::-1]
     
    self.scores, self.pred = self.softmax_layer.connect(self.inputs[-1])
    self.pred0 = self.pred.reshape([self.x.shape[0], self.x.shape[1]]).dimshuffle(1, 0)
    
  def get_eval_function(self):  
    """ We should feed in non-dimshuffled inputs x0, mask0 and y0.
        Used for tracking Dev loss at training time.
    """
    loss = CrossEntropyLoss().connect(self.scores, self.mask, self.y)
    return theano.function([self.x0, self.mask0, self.y0], [self.pred0, loss],
                 name='f_eval',
                 allow_input_downcast=True,
                 on_unused_input='warn',
                 givens=({self.is_train:  numpy.cast['int8'](0)}))
    
  def get_distribution_function(self):
    """ Return predictions and scores of shape [batch_size, time_steps, label space size].
        Used at test time.
    """
    scores0 = self.scores.reshape([self.x.shape[0], self.x.shape[1],
                     self.label_space_size]).dimshuffle(1, 0, 2)
                      
    return theano.function([self.x0, self.mask0], [self.pred0, scores0],
                 name='f_pred',
                 allow_input_downcast=True,
                 on_unused_input='warn',
                 givens=({self.is_train:  numpy.cast['int8'](0)}))
  
  def get_loss_function(self):
    """ We should feed in non-dimshuffled inputs x0, mask0 and y0.
    """
    loss = CrossEntropyLoss().connect(self.scores, self.mask, self.y)
    grads = gradient_clipping(tensor.grad(loss, self.params),
                  self.max_grad_norm)
    updates = adadelta(self.params, grads)

    return theano.function([self.x0, self.mask0, self.y0], loss,
                 name='f_loss',
                 updates=updates,
                 on_unused_input='warn',
                 givens=({self.is_train: numpy.cast['int8'](1)}))
  
  def save(self, filepath):
    """ Save model parameters to file.
    """
    all_params = OrderedDict([(param.name, param.get_value()) for param in self.params])
    numpy.savez(filepath, **all_params)
    print('Saved model to: {}'.format(filepath))
    

  def load(self, filepath):
    """ Load model parameters from file.
    """
    all_params = numpy.load(filepath)
    for param in self.params:
      if param.name in all_params:
        vals = all_params[param.name]
        if param.name.startswith('embedding') and self.embedding_shapes[0][0] > vals.shape[0]:
          # Expand to new vocabulary.
          print self.embedding_shapes[0][0], vals.shape[0]
          new_vals = numpy.concatenate((vals, param.get_value()[vals.shape[0]:, :]), axis=0)
          param.set_value(new_vals)
        else:
          param.set_value(vals)
    print('Loaded model from: {}'.format(filepath))
