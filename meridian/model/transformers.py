# Copyright 2024 The Meridian Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains data transformers for various inputs of the Meridian model."""

import numpy as np
import tensorflow as tf


__all__ = [
    "MediaTransformer",
    "ControlsTransformer",
    "KpiTransformer",
]


class MediaTransformer:
  """Contains forward and inverse media transformation methods.

  This class stores scale factors computed from per-geo medians of the `media`
  data, normalized by the geo population.
  """

  def __init__(
      self,
      media: tf.Tensor,
      population: tf.Tensor,
  ):
    """`MediaTransformer` constructor.

    Args:
      media: A tensor of dimension `(n_geos, n_media_times, n_media_channels)`
        containing the media data, used to compute the scale factors.
      population: A tensor of dimension `(n_geos,)` containing the population of
        each geo, used to compute the scale factors.
    """
    population_scaled_media = tf.math.divide_no_nan(
        media, population[:, tf.newaxis, tf.newaxis]
    )
    # Tensor of medians of the positive portion of `media`. Used as a component
    # for scaling.
    self._population_scaled_median_m = np.nanmedian(
        tf.where(population_scaled_media == 0, np.nan, population_scaled_media),
        [0, 1],
    )
    # Tensor of dimensions (`n_geos` x 1) of weights for scaling `metric`.
    self._scale_factors_gm = tf.einsum(
        "g,m->gm", population, self._population_scaled_median_m
    )

  @property
  def population_scaled_median_m(self):
    return self._population_scaled_median_m

  @tf.function(jit_compile=True)
  def forward(self, media: tf.Tensor) -> tf.Tensor:
    """Scales a given `media` tensor using the stored scale factors."""
    return media / self._scale_factors_gm[:, tf.newaxis, :]

  @tf.function(jit_compile=True)
  def inverse(self, media: tf.Tensor) -> tf.Tensor:
    """Scales a given `media` tensor using the inversed stored scale factors."""
    return media * self._scale_factors_gm[:, tf.newaxis, :]


class ControlsTransformer:
  """Contains forward and inverse controls transformation methods.

  This class stores means and standard deviations of the controls data, used
  to scale a given `controls` tensor.
  """

  def __init__(
      self,
      controls: tf.Tensor,
      population: tf.Tensor,
      population_scaling_id: tf.Tensor | None = None,
  ):
    """`ControlsTransformer` constructor.

    Args:
      controls: A tensor of dimension `(n_geos, n_times, n_controls)` containing
        the controls data, used to compute the mean and stddev.
      population: A tensor of dimension `(n_geos,)` containing the population of
        each geo, used to compute the scale factors.
      population_scaling_id: An optional boolean tensor of dimension
        `(n_controls,)` indicating the control variables for which the control
        value will be scaled by population.
    """
    if population_scaling_id is not None:
      self._population_scaling_factors = tf.where(
          population_scaling_id,
          population[:, None],
          tf.ones_like(population)[:, None],
      )
    else:
      self._population_scaling_factors = None

    self._means = tf.reduce_mean(controls, axis=(0, 1))
    self._stdevs = tf.math.reduce_std(controls, axis=(0, 1))

  @tf.function(jit_compile=True)
  def forward(self, controls: tf.Tensor) -> tf.Tensor:
    """Scales a given `controls` tensor using the stored coefficients."""
    if self._population_scaling_factors is not None:
      controls /= self._population_scaling_factors[:, None, :]
    return tf.math.divide_no_nan(controls - self._means, self._stdevs)

  @tf.function(jit_compile=True)
  def inverse(self, controls: tf.Tensor) -> tf.Tensor:
    """Scales back a given `controls` tensor using the stored coefficients."""
    scaled_controls = controls * self._stdevs + self._means
    return (
        scaled_controls * self._population_scaling_factors[:, None, :]
        if self._population_scaling_factors is not None
        else scaled_controls
    )


class KpiTransformer:
  """Contains forward and inverse KPI transformation methods.

  This class stores coefficients to scale KPI, first by geo and then
  by mean and standard deviation of KPI.
  """

  def __init__(
      self,
      kpi: tf.Tensor,
      population: tf.Tensor,
  ):
    """`KpiTransformer` constructor.

    Args:
      kpi: A tensor of dimension `(n_geos, n_times)` containing the KPI data,
        used to compute the mean and stddev.
      population: A tensor of dimension `(n_geos,)` containing the population of
        each geo, used to to compute the population scale factors.
    """
    self._population = population
    population_scaled_kpi = tf.math.divide_no_nan(
        kpi, self._population[:, tf.newaxis]
    )
    self._population_scaled_mean = tf.reduce_mean(population_scaled_kpi)
    self._population_scaled_stdev = tf.math.reduce_std(population_scaled_kpi)

  @tf.function(jit_compile=True)
  def forward(self, kpi: tf.Tensor) -> tf.Tensor:
    """Scales a given `kpi` tensor using the stored coefficients."""
    return tf.math.divide_no_nan(
        tf.math.divide_no_nan(kpi, self._population[:, tf.newaxis])
        - self._population_scaled_mean,
        self._population_scaled_stdev,
    )

  @tf.function(jit_compile=True)
  def inverse(self, kpi: tf.Tensor) -> tf.Tensor:
    """Scales back a given `kpi` tensor using the stored coefficients."""
    return (
        kpi * self._population_scaled_stdev + self._population_scaled_mean
    ) * self._population[:, tf.newaxis]
