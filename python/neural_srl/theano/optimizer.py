import numpy
import theano
import theano.tensor as tensor
from itertools import izip

from util import floatX


def adadelta(parameters, gradients, rho=0.95, eps=1e-6):
  """ Reference: ADADELTA: An Adaptive Learning Rate Method,
        Zeiler 2012. https://arxiv.org/abs/1212.5701
      Adapted from the Adadelta implementation from Tensorflow.
  """
  accum = [theano.shared(numpy.zeros(p.get_value().shape, floatX)) for p in parameters]
  accum_updates = [theano.shared(numpy.zeros(p.get_value().shape, floatX)) for p in parameters]
  
  new_accum = [rho * g0 + (1.0 - rho) * (g**2) for g0, g in izip(accum, gradients)]
  updates = [tensor.sqrt(d0 + eps) / tensor.sqrt(g0 + eps) * g for d0, g0, g in izip(accum_updates,
                                             new_accum,
                                             gradients)]
  
  new_accum_updates = [rho * d0 + (1.0 - rho) * (d**2) for d0, d in izip(accum_updates,
                                       updates)]
  
  accum_ = zip(accum, new_accum)
  accum_updates_ = zip(accum_updates, new_accum_updates)  
  parameters_ = [ (p, (p - d)) for p,d in izip(parameters, updates)]
  return accum_ + accum_updates_ + parameters_ 


def gradient_clipping(gradients, max_norm=5.0):
  global_grad_norm = tensor.sqrt(sum(map(lambda x: tensor.sqr(x).sum(), gradients)))
  multiplier = tensor.switch(global_grad_norm < max_norm, 1.0, max_norm / global_grad_norm)
  return [g * multiplier for g in gradients]
