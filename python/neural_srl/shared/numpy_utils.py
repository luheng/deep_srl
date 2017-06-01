import math
import numpy

def orth_normal_initializer(factor=1.0, seed=None):
  ''' Reference: Exact solutions to the nonlinear dynamics of learning in
                 deep linear neural networks
        Saxe et al., 2014. https://arxiv.org/pdf/1312.6120.pdf
      Adapted from the original implementation by Mingxuan Wang.
  '''
  def _initializer(shape, dtype):
    assert len(shape) == 2
    rng = numpy.random.RandomState(seed)
    if shape[0] == shape[1]:
      M = rng.randn(*shape).astype(dtype)
      Q, R = numpy.linalg.qr(M)
      Q = Q * numpy.sign(numpy.diag(R))
      param = Q * factor
      return param
    else:
      M1 = rng.randn(shape[0], shape[0]).astype(dtype)
      M2 = rng.randn(shape[1], shape[1]).astype(dtype)
      Q1, R1 = numpy.linalg.qr(M1)
      Q2, R2 = numpy.linalg.qr(M2)
      Q1 = Q1 * numpy.sign(numpy.diag(R1))
      Q2 = Q2 * numpy.sign(numpy.diag(R2))
      n_min = min(shape[0], shape[1])
      param = numpy.dot(Q1[:, :n_min], Q2[:n_min, :]) * factor
      return param
  return _initializer

def block_orth_normal_initializer(input_shapes, output_shapes, factor=1.0, seed=None):
  ''' Initialize a gigantic weight matrix where each block is a normal orthogonal matrix.
    Input:
      - input_shapes: the sizes of each block alone dimension 0.
      - output_shapes: the sizes of each block along dimension 1.
      for example input_shapes = [100, 128] output_shapes=[100,100,100,100]
        indicates eight blocks with shapes [100,100], [128,100], etc.
  '''
  def _initializer(shape, dtype):
    assert len(shape) == 2
    initializer = orth_normal_initializer(factor, seed)
    params = numpy.concatenate([numpy.concatenate([initializer([dim_in, dim_out], dtype)
             for dim_out in output_shapes], 1)
            for dim_in in input_shapes], 0)
    return params
    
  return _initializer

def random_normal_initializer(mean=0.0, stddev=0.01, seed=None):
  def _initializer(shape, dtype):
    rng = numpy.random.RandomState(seed)
    return numpy.asarray(rng.normal(mean, stddev, shape), dtype)

  return _initializer

def all_zero_initializer():
  def _initializer(shape, dtype):
    return numpy.zeros(shape).astype(dtype)

  return _initializer

def uniform_initializer(value=0.01):
  def _initializer(shape, dtype):
    return numpy.full(shape, value).astype(dtype)

  return _initializer
