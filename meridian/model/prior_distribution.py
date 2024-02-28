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

"""This file contains an object to store prior distributions.

The PriorDistribution object contains distributions for various parameters used
by the Meridian object.
"""
from collections.abc import MutableMapping
import dataclasses
from typing import Any
import warnings
from meridian import constants
import tensorflow_probability as tfp


@dataclasses.dataclass(frozen=True)
class PriorDistribution:
  """Contains prior distributions for each model parameter.

  PriorDistribution is a utility class for Meridian. The required shapes of the
  arguments to PriorDistribution depend on modeling options and data shapes
  passed to Meridian. For example, `ec_m` is a parameter that represents the
  half-saturation for each media channel. Thus, the `ec_m` argument must have
  either `batch_shape=[]` or `batch_shape` equal to the number of media
  channels. In the case of the former, each media channel gets the same prior.

  An error will be raised upon Meridian construction if any prior distribution
  has shape that cannot be broadcast to the shape designated by the model
  specification.

  The parameter batch shapes are as follows:
    knot_values: `n_knots`
    tau_g_excl_baseline: `n_geos - 1`
    beta_m: `n_media_channels`
    beta_rf: `n_rf_channels`
    eta_m: `n_media_channels`
    eta_rf: `n_rf_channels`
    gamma_c: `n_controls`
    xi_c: `n_controls`
    alpha_m: `n_media_channels`
    alpha_rf: `n_rf_channels`
    ec_m: `n_media_channels`
    ec_rf: `n_rf_channels`
    slope_m: `n_media_channels`
    slope_rf: `n_rf_channels`
    sigma: `n_geos` if `unique_sigma_for_each_geo` else 1
    roi_m: `n_media_channels`
    roi_rf: `n_rf_channels`

  Attributes:
    knot_values: prior distribution on knots for time effects. Default
      distribution is Normal(0.0, 5.0).
    tau_g_excl_baseline: prior distribution on geo effects which represent the
      average KPI of each geo relative to the baseline geo. This parameter is
      broadcast to a vector of length `n_geos - 1`, preserving the geo order and
      excluding the `baseline_geo`. After sampling, `Meridian.inference_data`
      includes a modified version of this parameter simply called `tau_g` which
      has length `n_geos` and contains a zero in the position corresponding to
      `baseline_geo`. Meridian ignores this distribution if `n_geos` = 1.
      Default distribution is Normal(0.0, 5.0).
    beta_m: prior distribution on a parameter for the hierarchical distribution
      of geo-level media effects for impression media channels (`beta_gm`). When
      `media_effects_dist` is set to "normal", it is the hierarchical mean. When
      `media_effects_dist` is set to "lognormal", it is the hierarchical
      parameter for the mean of the underlying, log-transformed, Normal
      distribution. Meridian ignores this distribution if `use_roi_prior` is
      `True` and uses the `roi_m` prior instead. Default distribution is
      HalfNormal(5.0).
    beta_rf: prior distribution on a parameter for the hierarchical distribution
      of geo-level media effects for reach and frequency media channels
      (`beta_grf`). When `media_effects_dist` is set to "normal", it is the
      hierarchical mean. When `media_effects_dist` is set to "lognormal", it is
      the hierarchical parameter for the mean of the underlying,
      log-transformed, Normal distribution. Meridian ignores this distribution
      if `use_roi_prior` is `True` and uses the `roi_m` prior instead. Default
      distribution is HalfNormal(5.0).
    eta_m: prior distribution on a parameter for the hierarchical distribution
      of geo-level media effects for impression media channels (beta_gm). When
      `media_effects_dist` is set to "normal", it is the hierarchical standard
      deviation. When `media_effects_dist` is set to "lognormal" it is the
      hierarchical parameter for the standard deviation of the underlying,
      log-transformed, Normal distribution. Default distribution is
      HalfNormal(1.0).
    eta_rf: prior distribution on a parameter for the hierarchical distribution
      of geo-level media effects for RF media channels (beta_grf). When
      `media_effects_dist` is set to "normal", it is the hierarchical standard
      deviation. When `media_effects_dist` is set to "lognormal" it is the
      hierarchical parameter for the standard deviation of the underlying,
      log-transformed, Normal distribution. Default distribution is
      HalfNormal(1.0).
    gamma_c: prior distribution on hierarchical mean of gamma_gc which is the
      coefficient on control c for geo g. Hierarchy is defined over geos.
      Default distribution is Normal(0.0, 5.0).
    xi_c: prior distribution on hierarchical standard deviation of gamma_gc
      which is the coefficient on control c for geo g. Hierarchy is defined over
      geos. Default distribution is HalfNormal(5.0).
    alpha_m: prior distribution on the geometric decay Adstock parameter for
      media input. Default distribution is Uniform(0.0, 1.0).
    alpha_rf: prior distribution on the geometric decay Adstock parameter for rf
      input. Default distribution is Uniform(0.0, 1.0).
    ec_m: prior distribution on the half-saturation Hill parameter for media
      input. Default distribution is TruncatedNormal(0.8, 0.8, 0.1, 10).
    ec_rf: prior distribution on the half-saturation Hill parameter for rf
      input. Default distribution is TransformedDistribution(LogNormal(0.7,
      0.4),Shift(0.1)).
    slope_m: prior distribution on the `slope` Hill parameter for media input.
      Default distribution is Deterministic(1.0).
    slope_rf: prior distribution on the `slope` Hill parameter for rf input.
      Default distribution is LogNormal(0.7, 0.4).
    sigma: prior distribution on the standard deviation of noise. Default
      distribution is HalfNormal(5.0).
    roi_m: prior distribution on hierarchical ROI in media input. Meridian
      ignores this distribution if `use_roi_prior` is `False` and uses `beta_m`
      instead. When `use_roi_prior` is `True` then `beta_m` is calculated as a
      deterministic function of `roi_m`, `alpha_m`, `ec_m`, `slope_m` and the
      spend associated with each media channel (i.e., the model is
      reparameterized with `roi_m` in place of `beta_m`). Default distribution
      is LogNormal(0.2, 0.9).
    roi_rf: prior distribution on hierarchical ROI in rf input. Meridian ignores
      this distribution if `use_roi_prior` is `False` and uses `beta_rf`
      instead. When `use_roi_prior` is `True` then `beta_rf` is calculated as a
      deterministic function of `roi_rf`, `alpha_rf`, `ec_rf`, `slope_rf` and
      the spend associated with each media channel (i.e., the model is
      reparameterized with `roi_rf` in place of `beta_rf`). Default distribution
      is LogNormal(0.2, 0.9).
  """

  knot_values: tfp.distributions.Distribution = tfp.distributions.Normal(
      0.0, 5.0, name=constants.KNOT_VALUES
  )
  tau_g_excl_baseline: tfp.distributions.Distribution = (
      tfp.distributions.Normal(0.0, 5.0, name=constants.TAU_G_EXCL_BASELINE)
  )
  beta_m: tfp.distributions.Distribution = tfp.distributions.HalfNormal(
      5.0, name=constants.BETA_M
  )
  beta_rf: tfp.distributions.Distribution = tfp.distributions.HalfNormal(
      5.0, name=constants.BETA_RF
  )
  eta_m: tfp.distributions.Distribution = tfp.distributions.HalfNormal(
      1.0, name=constants.ETA_M
  )
  eta_rf: tfp.distributions.Distribution = tfp.distributions.HalfNormal(
      1.0, name=constants.ETA_RF
  )
  gamma_c: tfp.distributions.Distribution = tfp.distributions.Normal(
      0.0, 5.0, name=constants.GAMMA_C
  )
  xi_c: tfp.distributions.Distribution = tfp.distributions.HalfNormal(
      5.0, name=constants.XI_C
  )
  alpha_m: tfp.distributions.Distribution = tfp.distributions.Uniform(
      0.0, 1.0, name=constants.ALPHA_M
  )
  alpha_rf: tfp.distributions.Distribution = tfp.distributions.Uniform(
      0.0, 1.0, name=constants.ALPHA_RF
  )
  ec_m: tfp.distributions.Distribution = tfp.distributions.TruncatedNormal(
      0.8, 0.8, 0.1, 10, name=constants.EC_M
  )
  ec_rf: tfp.distributions.Distribution = (
      tfp.distributions.TransformedDistribution(
          tfp.distributions.LogNormal(0.7, 0.4),
          tfp.bijectors.Shift(0.1),
          name=constants.EC_RF,
      )
  )
  slope_m: tfp.distributions.Distribution = tfp.distributions.Deterministic(
      1.0, name=constants.SLOPE_M
  )
  slope_rf: tfp.distributions.Distribution = tfp.distributions.LogNormal(
      0.7, 0.4, name=constants.SLOPE_RF
  )
  sigma: tfp.distributions.Distribution = tfp.distributions.HalfNormal(
      5.0, name=constants.SIGMA
  )
  roi_m: tfp.distributions.Distribution = tfp.distributions.LogNormal(
      0.2, 0.9, name=constants.ROI_M
  )
  roi_rf: tfp.distributions.Distribution = tfp.distributions.LogNormal(
      0.2, 0.9, name=constants.ROI_RF
  )

  def has_deterministic_param(
      self, param: tfp.distributions.Distribution
  ) -> bool:
    return hasattr(self, param) and isinstance(
        getattr(self, param).distribution, tfp.distributions.Deterministic
    )

  def __setstate__(self, state):
    # Override to support pickling.

    def _unpack_distribution_params(
        params: MutableMapping[str, Any]
    ) -> tfp.distributions.Distribution:
      if constants.DISTRIBUTION in params:
        params[constants.DISTRIBUTION] = _unpack_distribution_params(
            params[constants.DISTRIBUTION]
        )
      dist_type = params.pop(constants.DISTRIBUTION_TYPE)
      return dist_type(**params)

    new_state = {}
    for attribute, value in state.items():
      new_state[attribute] = _unpack_distribution_params(value)

    self.__dict__.update(new_state)

  def __getstate__(self):
    # Override to support pickling.
    state = self.__dict__.copy()

    def _pack_distribution_params(
        dist: tfp.distributions.Distribution,
    ) -> MutableMapping[str, Any]:
      params = dist.parameters
      params[constants.DISTRIBUTION_TYPE] = type(dist)
      if constants.DISTRIBUTION in params:
        params[constants.DISTRIBUTION] = _pack_distribution_params(
            dist.distribution
        )
      return params

    for attribute, value in state.items():
      state[attribute] = _pack_distribution_params(value)

    return state


def _convert_to_deterministic_0_distribution(
    distribution: tfp.distributions.Distribution,
) -> tfp.distributions.Distribution:
  """Converts the given `value` to Deterministic(0) distribution.

  Args:
    distribution: tfp.distributions.Distribution object to be converted to
      Deterministic(0) distribution.

  Returns:
    tfp.distribution.Deterministic(0, distribution.name)

  Raises:
    Warning: If the argument distribution is not the Deterministic(0)
    distribution.
  """
  if (
      not isinstance(distribution, tfp.distributions.Deterministic)
      or distribution.loc != 0
  ):
    warnings.warn(
        'Hierarchical distribution parameters must be deterministically zero'
        ' for national models. {} has been automatically set to'
        ' Deterministic(0).'.format(distribution.name)
    )
    return tfp.distributions.Deterministic(loc=0, name=distribution.name)
  else:
    return distribution


def broadcast_prior_distribution(
    prior_distribution: PriorDistribution,
    n_geos: int,
    n_media_channels: int,
    n_rf_channels: int,
    n_controls: int,
    sigma_shape: int,
    n_knots: int,
    is_national: bool,
) -> PriorDistribution:
  """Broadcast distribution attributes.

  Args:
    prior_distribution: The prior distribution to be broadcast.
    n_geos: Number of geos.
    n_media_channels: Number of media channels used.
    n_rf_channels: Number of rf channels used.
    n_controls: Number of controls used.
    sigma_shape: A number describing the shape of the sigma parameter. It's
      either 1 (if sigma_for_each_geo=False) or `n_geos` (if
      sigma_for_each_geo=True), see `ModelSpec`.
    n_knots: Number of knots used.
    is_national: A boolean indicator whether prior distribution should be
      adapted for a national model.

  Returns:
    Prior distribution broadcast according to the data dimensionality.
  """
  # Broadcast Distributions.
  knot_values = tfp.distributions.BatchBroadcast(
      prior_distribution.knot_values,
      n_knots,
      name=constants.KNOT_VALUES,
  )
  if is_national:
    tau_g_converted = _convert_to_deterministic_0_distribution(
        prior_distribution.tau_g_excl_baseline
    )
  else:
    tau_g_converted = prior_distribution.tau_g_excl_baseline
  tau_g_excl_baseline = tfp.distributions.BatchBroadcast(
      tau_g_converted, n_geos - 1, name=constants.TAU_G_EXCL_BASELINE
  )
  beta_m = tfp.distributions.BatchBroadcast(
      prior_distribution.beta_m, n_media_channels, name=constants.BETA_M
  )
  beta_rf = tfp.distributions.BatchBroadcast(
      prior_distribution.beta_rf, n_rf_channels, name=constants.BETA_RF
  )
  if is_national:
    eta_m_converted = _convert_to_deterministic_0_distribution(
        prior_distribution.eta_m
    )
    eta_rf_converted = _convert_to_deterministic_0_distribution(
        prior_distribution.eta_rf
    )
  else:
    eta_m_converted = prior_distribution.eta_m
    eta_rf_converted = prior_distribution.eta_rf
  eta_m = tfp.distributions.BatchBroadcast(
      eta_m_converted, n_media_channels, name=constants.ETA_M
  )
  eta_rf = tfp.distributions.BatchBroadcast(
      eta_rf_converted, n_rf_channels, name=constants.ETA_RF
  )
  gamma_c = tfp.distributions.BatchBroadcast(
      prior_distribution.gamma_c, n_controls, name=constants.GAMMA_C
  )
  if is_national:
    xi_c_converted = _convert_to_deterministic_0_distribution(
        prior_distribution.xi_c
    )
  else:
    xi_c_converted = prior_distribution.xi_c
  xi_c = tfp.distributions.BatchBroadcast(
      xi_c_converted, n_controls, name=constants.XI_C
  )
  alpha_m = tfp.distributions.BatchBroadcast(
      prior_distribution.alpha_m, n_media_channels, name=constants.ALPHA_M
  )
  alpha_rf = tfp.distributions.BatchBroadcast(
      prior_distribution.alpha_rf, n_rf_channels, name=constants.ALPHA_RF
  )
  ec_m = tfp.distributions.BatchBroadcast(
      prior_distribution.ec_m, n_media_channels, name=constants.EC_M
  )
  ec_rf = tfp.distributions.BatchBroadcast(
      prior_distribution.ec_rf, n_rf_channels, name=constants.EC_RF
  )
  slope_m = tfp.distributions.BatchBroadcast(
      prior_distribution.slope_m, n_media_channels, name=constants.SLOPE_M
  )
  slope_rf = tfp.distributions.BatchBroadcast(
      prior_distribution.slope_rf, n_rf_channels, name=constants.SLOPE_RF
  )
  sigma = tfp.distributions.BatchBroadcast(
      prior_distribution.sigma, sigma_shape, name=constants.SIGMA
  )
  roi_m = tfp.distributions.BatchBroadcast(
      prior_distribution.roi_m, n_media_channels, name=constants.ROI_M
  )
  roi_rf = tfp.distributions.BatchBroadcast(
      prior_distribution.roi_rf, n_rf_channels, name=constants.ROI_RF
  )

  return PriorDistribution(
      knot_values=knot_values,
      tau_g_excl_baseline=tau_g_excl_baseline,
      beta_m=beta_m,
      beta_rf=beta_rf,
      eta_m=eta_m,
      eta_rf=eta_rf,
      gamma_c=gamma_c,
      xi_c=xi_c,
      alpha_m=alpha_m,
      alpha_rf=alpha_rf,
      ec_m=ec_m,
      ec_rf=ec_rf,
      slope_m=slope_m,
      slope_rf=slope_rf,
      sigma=sigma,
      roi_m=roi_m,
      roi_rf=roi_rf,
  )
