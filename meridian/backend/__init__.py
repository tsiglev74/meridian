"""Backend abstraction layer for tensor operations and probability distributions.

This module provides a unified interface for tensor operations and probability
distributions, allowing the codebase to switch between TensorFlow and JAX
implementations.
"""

from typing import Any, Optional, Sequence, TypeVar, Union

import numpy as np

# Import the default backend (TensorFlow)
from meridian.backend.tensorflow import backend as tf_backend

# Re-export backend functions and types
Tensor = tf_backend.Tensor
Distribution = tf_backend.Distribution
Bijector = tf_backend.Bijector

# Re-export backend functions
cast = tf_backend.cast
concat = tf_backend.concat
einsum = tf_backend.einsum
expand_dims = tf_backend.expand_dims
math = tf_backend.math
random = tf_backend.random
transpose = tf_backend.transpose
zeros = tf_backend.zeros
zeros_like = tf_backend.zeros_like

# Re-export distribution functions
deterministic = tf_backend.deterministic
half_normal = tf_backend.half_normal
log_normal = tf_backend.log_normal
normal = tf_backend.normal
sample = tf_backend.sample
truncated_normal = tf_backend.truncated_normal
uniform = tf_backend.uniform

# Re-export bijector functions
shift = tf_backend.shift

# Re-export MCMC functions
windowed_adaptive_nuts = tf_backend.windowed_adaptive_nuts

# Type hints
T = TypeVar('T')

def convert_to_tensor(value: Any, dtype: Optional[Any] = None) -> Tensor:
  """Convert a value to a tensor."""
  return tf_backend.convert_to_tensor(value, dtype)

def boolean_mask(tensor: Tensor, mask: Tensor, axis: int) -> Tensor:
  """Apply boolean mask to tensor along specified axis."""
  return tf_backend.boolean_mask(tensor, mask, axis)

def reduce_sum(tensor: Tensor, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> Tensor:
  """Compute sum of tensor elements along specified axes."""
  return tf_backend.reduce_sum(tensor, axis, keepdims)

def function(jit_compile: bool = True, **kwargs):
  """Decorator for JIT compilation."""
  return tf_backend.function(jit_compile=jit_compile, **kwargs)

def sanitize_seed(seed: Optional[Union[int, Sequence[int]]]) -> Optional[Union[int, Sequence[int]]]:
  """Sanitize random seed."""
  return tf_backend.sanitize_seed(seed)

def broadcast_to(tensor: Tensor, shape: Sequence[int]) -> Tensor:
  """Broadcast tensor to specified shape."""
  return tf_backend.broadcast_to(tensor, shape)

def newaxis() -> int:
  """Return index for new axis."""
  return tf_backend.newaxis()

def test_case():
  """Return test case class for current backend."""
  return tf_backend.test_case()

def experimental_extension_type():
  """Return extension type class for current backend."""
  return tf_backend.experimental_extension_type()
