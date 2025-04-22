"""TensorFlow backend implementation."""

from typing import Any, Optional, Sequence, TypeVar, Union

import tensorflow as tf
import tensorflow_probability as tfp

# Type aliases
Tensor = tf.Tensor
Distribution = tfp.distributions.Distribution
Bijector = tfp.bijectors.Bijector

# Tensor operations
cast = tf.cast
concat = tf.concat
einsum = tf.einsum
expand_dims = tf.expand_dims
math = tf.math
random = tf.random
transpose = tf.transpose
zeros = tf.zeros
zeros_like = tf.zeros_like

# Distribution functions
deterministic = tfp.distributions.Deterministic
half_normal = tfp.distributions.HalfNormal
log_normal = tfp.distributions.LogNormal
normal = tfp.distributions.Normal
sample = tfp.distributions.Sample
truncated_normal = tfp.distributions.TruncatedNormal
uniform = tfp.distributions.Uniform

# Bijector functions
shift = tfp.bijectors.Shift

# MCMC functions
windowed_adaptive_nuts = tfp.experimental.mcmc.windowed_adaptive_nuts

def convert_to_tensor(value: Any, dtype: Optional[Any] = None) -> Tensor:
  """Convert a value to a tensor."""
  return tf.convert_to_tensor(value, dtype)

def boolean_mask(tensor: Tensor, mask: Tensor, axis: int) -> Tensor:
  """Apply boolean mask to tensor along specified axis."""
  return tf.boolean_mask(tensor, mask, axis)

def reduce_sum(tensor: Tensor, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> Tensor:
  """Compute sum of tensor elements along specified axes."""
  return tf.reduce_sum(tensor, axis, keepdims)

def function(jit_compile: bool = True, **kwargs):
  """Decorator for JIT compilation."""
  return tf.function(jit_compile=jit_compile, **kwargs)

def sanitize_seed(seed: Optional[Union[int, Sequence[int]]]) -> Optional[Union[int, Sequence[int]]]:
  """Sanitize random seed."""
  return tfp.random.sanitize_seed(seed)

def broadcast_to(tensor: Tensor, shape: Sequence[int]) -> Tensor:
  """Broadcast tensor to specified shape."""
  return tf.broadcast_to(tensor, shape)

def newaxis() -> int:
  """Return index for new axis."""
  return tf.newaxis

def test_case():
  """Return test case class for current backend."""
  return tf.test.TestCase

def experimental_extension_type():
  """Return extension type class for current backend."""
  return tf.experimental.ExtensionType
