"""Tests for the backend abstraction layer."""

from absl.testing import absltest
import numpy as np
from meridian.backend import (
    Tensor,
    Distribution,
    cast,
    concat,
    einsum,
    expand_dims,
    math,
    transpose,
    zeros,
    zeros_like,
    deterministic,
    half_normal,
    log_normal,
    normal,
    sample,
    truncated_normal,
    uniform,
    shift,
    convert_to_tensor,
    boolean_mask,
    reduce_sum,
    function,
    sanitize_seed,
    broadcast_to,
    newaxis,
    test_case,
    experimental_extension_type,
)


class BackendTest(test_case(), absltest.TestCase):
  """Tests for the backend abstraction layer."""

  def test_tensor_operations(self):
    """Test basic tensor operations."""
    # Create tensors
    a = convert_to_tensor([1, 2, 3])
    b = convert_to_tensor([4, 5, 6])

    # Test operations
    c = concat([a, b], axis=0)
    self.assertAllEqual(c, [1, 2, 3, 4, 5, 6])

    d = einsum("i,j->ij", a, b)
    self.assertAllEqual(d, [[4, 5, 6], [8, 10, 12], [12, 15, 18]])

    e = expand_dims(a, axis=0)
    self.assertAllEqual(e, [[1, 2, 3]])

    f = transpose(d)
    self.assertAllEqual(f, [[4, 8, 12], [5, 10, 15], [6, 12, 18]])

    g = zeros([2, 3])
    self.assertAllEqual(g, [[0, 0, 0], [0, 0, 0]])

    h = zeros_like(a)
    self.assertAllEqual(h, [0, 0, 0])

    i = cast(a, float)
    self.assertAllEqual(i, [1.0, 2.0, 3.0])

    j = boolean_mask(a, [True, False, True], axis=0)
    self.assertAllEqual(j, [1, 3])

    k = reduce_sum(d)
    self.assertAllEqual(k, 90)

    l = broadcast_to(a, [2, 3])
    self.assertAllEqual(l, [[1, 2, 3], [1, 2, 3]])

  def test_distributions(self):
    """Test probability distributions."""
    # Test deterministic distribution
    d = deterministic(1.0)
    self.assertAllEqual(d.sample(), 1.0)

    # Test normal distribution
    n = normal(0.0, 1.0)
    samples = n.sample(1000)
    self.assertAllClose(np.mean(samples), 0.0, atol=0.1)
    self.assertAllClose(np.std(samples), 1.0, atol=0.1)

    # Test half normal distribution
    hn = half_normal(1.0)
    samples = hn.sample(1000)
    self.assertAllGreater(samples, 0.0)

    # Test log normal distribution
    ln = log_normal(0.0, 1.0)
    samples = ln.sample(1000)
    self.assertAllGreater(samples, 0.0)

    # Test truncated normal distribution
    tn = truncated_normal(0.0, 1.0, -1.0, 1.0)
    samples = tn.sample(1000)
    self.assertAllGreater(samples, -1.0)
    self.assertAllLess(samples, 1.0)

    # Test uniform distribution
    u = uniform(0.0, 1.0)
    samples = u.sample(1000)
    self.assertAllGreater(samples, 0.0)
    self.assertAllLess(samples, 1.0)

    # Test sample distribution
    s = sample(normal(0.0, 1.0), [2, 3])
    samples = s.sample()
    self.assertEqual(samples.shape, (2, 3))

  def test_bijectors(self):
    """Test bijectors."""
    # Test shift bijector
    s = shift(1.0)
    self.assertAllEqual(s.forward(0.0), 1.0)
    self.assertAllEqual(s.inverse(1.0), 0.0)

  def test_function_decorator(self):
    """Test function decorator."""
    @function(jit_compile=True)
    def add(a, b):
      return a + b

    result = add(1, 2)
    self.assertAllEqual(result, 3)

  def test_seed_sanitization(self):
    """Test seed sanitization."""
    seed = sanitize_seed(42)
    self.assertIsInstance(seed, int)

    seed = sanitize_seed([42, 43])
    self.assertIsInstance(seed, list)
    self.assertEqual(len(seed), 2)

  def test_extension_type(self):
    """Test extension type."""
    @experimental_extension_type()
    class TestType:
      def __init__(self, value):
        self.value = value

      def __validate__(self):
        pass

    t = TestType(42)
    self.assertEqual(t.value, 42)


if __name__ == "__main__":
  absltest.main()
