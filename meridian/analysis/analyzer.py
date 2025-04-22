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

"""Methods to compute analysis metrics of the model and the data."""

from collections.abc import Mapping, Sequence
import itertools
from typing import Any, Optional, Union
import warnings

from meridian import constants
from meridian.backend import (
    Tensor,
    cast,
    concat,
    einsum,
    expand_dims,
    math,
    transpose,
    zeros,
    zeros_like,
    boolean_mask,
    reduce_sum,
    function,
    broadcast_to,
    experimental_extension_type,
)
from meridian.model import adstock_hill
from meridian.model import model
from meridian.model import transformers
import numpy as np
import pandas as pd
from typing_extensions import Self
import xarray as xr

__all__ = [
    "Analyzer",
    "DataTensors",
    "DistributionTensors",
]


@experimental_extension_type()
class DataTensors:
  """Container for data variable arguments of Analyzer methods.

  Attributes:
    kpi: A tensor of shape `[n_geos, n_times]` containing the KPI values.
    media: A tensor of shape `[n_geos, n_times, n_media_channels]` containing
      the media values.
    media_spend: A tensor of shape `[n_geos, n_times, n_media_channels]`
      containing the media spend values.
    controls: A tensor of shape `[n_geos, n_times, n_controls]` containing the
      control values.
    non_media_treatments: A tensor of shape `[n_geos, n_times,
      n_non_media_channels]` containing the non-media treatment values.
    population: A tensor of shape `[n_geos, n_times]` containing the population
      values.
    time: A tensor of shape `[n_times]` containing the time values.
    geo: A tensor of shape `[n_geos]` containing the geo values.
    media_names: A sequence of strings containing the media channel names.
    control_names: A sequence of strings containing the control names.
    non_media_names: A sequence of strings containing the non-media channel
      names.
  """

  kpi: Tensor
  media: Optional[Tensor] = None
  media_spend: Optional[Tensor] = None
  controls: Optional[Tensor] = None
  non_media_treatments: Optional[Tensor] = None
  population: Optional[Tensor] = None
  time: Optional[Tensor] = None
  geo: Optional[Tensor] = None
  media_names: Optional[Sequence[str]] = None
  control_names: Optional[Sequence[str]] = None
  non_media_names: Optional[Sequence[str]] = None

  def __validate__(self):
    """Validates the data tensors."""
    if self.kpi is None:
      raise ValueError("kpi must be provided")
    if self.media is not None and self.media_spend is None:
      raise ValueError("media_spend must be provided if media is provided")
    if self.media is not None and self.media_names is None:
      raise ValueError("media_names must be provided if media is provided")
    if self.controls is not None and self.control_names is None:
      raise ValueError("control_names must be provided if controls is provided")
    if self.non_media_treatments is not None and self.non_media_names is None:
      raise ValueError(
          "non_media_names must be provided if non_media_treatments is provided"
      )


@experimental_extension_type()
class DistributionTensors:
  """Container for distribution variable arguments of Analyzer methods.

  Attributes:
    knot_values: A tensor of shape `[n_knots]` containing the knot values.
    tau_g: A tensor of shape `[n_geos]` containing the geo effects.
    beta_m: A tensor of shape `[n_media_channels]` containing the media
      effects.
    beta_rf: A tensor of shape `[n_rf_channels]` containing the reach and
      frequency effects.
    beta_om: A tensor of shape `[n_organic_media_channels]` containing the
      organic media effects.
    beta_orf: A tensor of shape `[n_organic_rf_channels]` containing the
      organic reach and frequency effects.
    gamma_c: A tensor of shape `[n_controls]` containing the control effects.
    gamma_n: A tensor of shape `[n_non_media_channels]` containing the
      non-media effects.
    sigma: A tensor of shape `[n_geos]` containing the noise standard
      deviations.
  """

  knot_values: Tensor
  tau_g: Tensor
  beta_m: Optional[Tensor] = None
  beta_rf: Optional[Tensor] = None
  beta_om: Optional[Tensor] = None
  beta_orf: Optional[Tensor] = None
  gamma_c: Optional[Tensor] = None
  gamma_n: Optional[Tensor] = None
  sigma: Tensor

  def __validate__(self):
    """Validates the distribution tensors."""
    if self.knot_values is None:
      raise ValueError("knot_values must be provided")
    if self.tau_g is None:
      raise ValueError("tau_g must be provided")
    if self.sigma is None:
      raise ValueError("sigma must be provided")


def _transformed_new_or_scaled(
    new_variable: Tensor | None,
    transformer: transformers.TensorTransformer | None,
    scaled_variable: Tensor | None,
) -> Tensor | None:
  """Returns the transformed new variable or the scaled variable.

  If the `new_variable` is present, returns
  `transformer.forward(new_variable)`. Otherwise, returns the
  `scaled_variable`.

  Args:
    new_variable: Optional tensor to be transformed..
    transformer: Optional DataTransformer.
    scaled_variable: Tensor to be returned if `new_variable` is None.

  Returns:
    The transformed new variable (if the new variable is present) or the
    original scaled variable from the input data otherwise.
  """
  if new_variable is None or transformer is None:
    return scaled_variable
  return transformer.forward(new_variable)


def get_central_tendency_and_ci(
    data: np.ndarray | Tensor,
    confidence_level: float = constants.DEFAULT_CONFIDENCE_LEVEL,
    axis: tuple[int, ...] = (0, 1),
    include_median=False,
) -> np.ndarray:
  """Calculates central tendency and confidence intervals for the given data.

  Args:
    data: Data for the metric.
    confidence_level: Confidence level for computing credible intervals,
      represented as a value between zero and one.
    axis: Axis or axes along which the mean, median, and quantiles are computed.
    include_median: A boolean flag indicating whether to calculate and include
      the median in the output Dataset (default: False).

  Returns:
    A numpy array or tf.Tensor containing central tendency and confidence
    intervals.
  """
  mean = np.mean(data, axis=axis, keepdims=False)
  ci_lo = np.quantile(data, (1 - confidence_level) / 2, axis=axis)
  ci_hi = np.quantile(data, (1 + confidence_level) / 2, axis=axis)

  if include_median:
    median = np.median(data, axis=axis, keepdims=False)
    return np.stack([mean, median, ci_lo, ci_hi], axis=-1)
  else:
    return np.stack([mean, ci_lo, ci_hi], axis=-1)


def _calc_rsquared(expected, actual):
  """Calculates r-squared between actual and expected outcome."""
  return 1 - np.nanmean((expected - actual) ** 2) / np.nanvar(actual)


def _calc_mape(expected, actual):
  """Calculates MAPE between actual and expected outcome."""
  return np.nanmean(np.abs((actual - expected) / actual))


def _calc_weighted_mape(expected, actual):
  """Calculates wMAPE between actual and expected outcome (weighted by actual)."""
  return np.nansum(np.abs(actual - expected)) / np.nansum(actual)


def _warn_if_geo_arg_in_kwargs(**kwargs):
  """Raises warning if a geo-level argument is used with national model."""
  for kwarg, value in kwargs.items():
    if (
        kwarg in constants.NATIONAL_ANALYZER_PARAMETERS_DEFAULTS
        and value != constants.NATIONAL_ANALYZER_PARAMETERS_DEFAULTS[kwarg]
    ):
      warnings.warn(
          f"The `{kwarg}` argument is ignored in the national model. It will be"
          " reset to"
          f" `{constants.NATIONAL_ANALYZER_PARAMETERS_DEFAULTS[kwarg]}`."
      )


def _check_n_dims(tensor: Tensor, name: str, n_dims: int):
  """Raises an error if the tensor has the wrong number of dimensions."""
  if tensor.ndim != n_dims:
    raise ValueError(
        f"New `{name}` must have {n_dims} dimension(s). Found"
        f" {tensor.ndim} dimension(s)."
    )


def _is_bool_list(l: Sequence[Any]) -> bool:
  """Returns True if the list contains only booleans."""
  return all(isinstance(item, bool) for item in l)


def _is_str_list(l: Sequence[Any]) -> bool:
  """Returns True if the list contains only strings."""
  return all(isinstance(item, str) for item in l)


def _validate_selected_times(
    selected_times: Sequence[str] | Sequence[bool],
    input_times: xr.DataArray,
    n_times: int,
    arg_name: str,
    comparison_arg_name: str,
):
  """Raises an error if selected_times is invalid.

  This checks that the `selected_times` argument is a list of strings or a list
  of booleans. If it is a list of strings, then each string must match the name
  of a time period coordinate in `input_times`. If it is a list of booleans,
  then it must have the same number of elements as `n_times`.

  Args:
    selected_times: Optional list of times to validate.
    input_times: Time dimension coordinates from `InputData.time` or
      `InputData.media_time`.
    n_times: The number of time periods in the tensor.
    arg_name: The name of the argument being validated.
    comparison_arg_name: The name of the argument being compared to.
  """
  if not selected_times:
    return
  if _is_bool_list(selected_times):
    if len(selected_times) != n_times:
      raise ValueError(
          f"Boolean `{arg_name}` must have the same number of elements as "
          f"there are time period coordinates in {comparison_arg_name}."
      )
  elif _is_str_list(selected_times):
    if any(time not in input_times for time in selected_times):
      raise ValueError(
          f"`{arg_name}` must match the time dimension names from "
          "meridian.InputData."
      )
  else:
    raise ValueError(
        f"`{arg_name}` must be a list of strings or a list of booleans."
    )


def _validate_flexible_selected_times(
    selected_times: Sequence[str] | Sequence[bool] | None,
    media_selected_times: Sequence[str] | Sequence[bool] | None,
    new_n_media_times: int,
):
  """Raises an error if selected times or media selected times is invalid.

  This checks that the `selected_times` and `media_selected_times` arguments
  are lists of booleans with the same number of elements as `new_n_media_times`.
  This is only relevant if the time dimension of any of the variables in
  `new_data` used in the analysis is modified.

  Args:
    selected_times: Optional list of times to validate.
    media_selected_times: Optional list of media times to validate.
    new_n_media_times: The number of time periods in the new data.
  """
  if selected_times and (
      not _is_bool_list(selected_times)
      or len(selected_times) != new_n_media_times
  ):
    raise ValueError(
        "If `media`, `reach`, `frequency`, `organic_media`,"
        " `organic_reach`, `organic_frequency`, `non_media_treatments`, or"
        " `revenue_per_kpi` is provided with a different number of time"
        " periods than in `InputData`, then `selected_times` must be a list"
        " of booleans with length equal to the number of time periods in"
        " the new data."
    )

  if media_selected_times and (
      not _is_bool_list(media_selected_times)
      or len(media_selected_times) != new_n_media_times
  ):
    raise ValueError(
        "If `media`, `reach`, `frequency`, `organic_media`,"
        " `organic_reach`, `organic_frequency`, `non_media_treatments`, or"
        " `revenue_per_kpi` is provided with a different number of time"
        " periods than in `InputData`, then `media_selected_times` must be"
        " a list of booleans with length equal to the number of time"
        " periods in the new data."
    )


def _scale_tensors_by_multiplier(
    data: DataTensors,
    multiplier: float,
    by_reach: bool,
    non_media_treatments_baseline: Tensor | None = None,
) -> DataTensors:
  """Get scaled tensors for incremental outcome calculation.

  Args:
    data: DataTensors object containing the optional tensors to scale. Only
      `media`, `reach`, `frequency`, `organic_media`, `organic_reach`,
      `organic_frequency`, `non_media_treatments` are scaled. The other tensors
      remain unchanged.
    multiplier: Float indicating the factor to scale tensors by.
    by_reach: Boolean indicating whether to scale reach or frequency when rf
      data is available.
    non_media_treatments_baseline: Optional tensor to overwrite
      `data.non_media_treatments` in the output. Used to compute the
      conterfactual values for incremental outcome calculation. If not used, the
      unmodified `data.non_media_treatments` tensor is returned in the output.

  Returns:
    A `DataTensors` object containing scaled tensor parameters. The original
    tensors that should not be scaled remain unchanged.
  """
  incremented_data = {}
  if data.media is not None:
    incremented_data[constants.MEDIA] = data.media * multiplier
  if data.reach is not None and data.frequency is not None:
    if by_reach:
      incremented_data[constants.REACH] = data.reach * multiplier
      incremented_data[constants.FREQUENCY] = data.frequency
    else:
      incremented_data[constants.REACH] = data.reach
      incremented_data[constants.FREQUENCY] = data.frequency * multiplier
  if data.organic_media is not None:
    incremented_data[constants.ORGANIC_MEDIA] = data.organic_media * multiplier
  if data.organic_reach is not None and data.organic_frequency is not None:
    if by_reach:
      incremented_data[constants.ORGANIC_REACH] = (
          data.organic_reach * multiplier
      )
      incremented_data[constants.ORGANIC_FREQUENCY] = data.organic_frequency
    else:
      incremented_data[constants.ORGANIC_REACH] = data.organic_reach
      incremented_data[constants.ORGANIC_FREQUENCY] = (
          data.organic_frequency * multiplier
      )
  if non_media_treatments_baseline is not None:
    incremented_data[constants.NON_MEDIA_TREATMENTS] = (
        non_media_treatments_baseline
    )
  else:
    incremented_data[constants.NON_MEDIA_TREATMENTS] = data.non_media_treatments

  # Include the original data that does not get scaled.
  incremented_data[constants.MEDIA_SPEND] = data.media_spend
  incremented_data[constants.RF_SPEND] = data.rf_spend
  incremented_data[constants.CONTROLS] = data.controls
  incremented_data[constants.REVENUE_PER_KPI] = data.revenue_per_kpi

  return DataTensors(**incremented_data)


def _central_tendency_and_ci_by_prior_and_posterior(
    prior: Tensor,
    posterior: Tensor,
    metric_name: str,
    xr_dims: Sequence[str],
    xr_coords: Mapping[str, tuple[Sequence[str], Sequence[str]]],
    confidence_level: float = constants.DEFAULT_CONFIDENCE_LEVEL,
    include_median: bool = False,
) -> xr.Dataset:
  """Calculates central tendency and CI of prior/posterior data for a metric.

  Args:
    prior: A tensor with the prior data for the metric.
    posterior: A tensor with the posterior data for the metric.
    metric_name: The name of the input metric for the computations.
    xr_dims: A list of dimensions for the output dataset.
    xr_coords: A dictionary with the coordinates for the output dataset.
    confidence_level: Confidence level for computing credible intervals,
      represented as a value between zero and one.
    include_median: A boolean flag indicating whether to calculate and include
      the median in the output Dataset (default: False).

  Returns:
    An xarray Dataset containing central tendency and confidence intervals for
    prior and posterior data for the metric.
  """
  metrics = np.stack(
      [
          get_central_tendency_and_ci(
              prior, confidence_level, include_median=include_median
          ),
          get_central_tendency_and_ci(
              posterior, confidence_level, include_median=include_median
          ),
      ],
      axis=-1,
  )
  xr_data = {metric_name: (xr_dims, metrics)}
  return xr.Dataset(data_vars=xr_data, coords=xr_coords)


def _compute_non_media_baseline(
    non_media_treatments: Tensor,
    non_media_baseline_values: Sequence[float | str] | None = None,
    non_media_selected_times: Sequence[bool] | None = None,
) -> Tensor:
  """Computes the baseline for each non-media treatment channel.

  Args:
    non_media_treatments: The non-media treatment input data.
    non_media_baseline_values: Optional list of shape (n_non_media_channels,).
      Each element is either a float (which means that the fixed value will be
      used as baseline for the given channel) or one of the strings "min" or
      "max" (which mean that the global minimum or maximum value will be used as
      baseline for the values of the given non_media treatment channel). If
      None, the minimum value is used as baseline for each non_media treatment
      channel.
    non_media_selected_times: Optional list of shape (n_times,). Each element is
      a boolean indicating whether the corresponding time period should be
      included in the baseline computation.

  Returns:
    A tensor of shape (n_geos, n_times, n_non_media_channels) containing the
    baseline values for each non-media treatment channel.
  """

  if non_media_selected_times is None:
    non_media_selected_times = [True] * non_media_treatments.shape[-2]

  if non_media_baseline_values is None:
    # If non_media_baseline_values is not provided, use the minimum value for
    # each non_media treatment channel as the baseline.
    non_media_baseline_values_filled = [
        constants.NON_MEDIA_BASELINE_MIN
    ] * non_media_treatments.shape[-1]
  else:
    non_media_baseline_values_filled = non_media_baseline_values

  if non_media_treatments.shape[-1] != len(non_media_baseline_values_filled):
    raise ValueError(
        "The number of non-media channels"
        f" ({non_media_treatments.shape[-1]}) does not match the number"
        f" of baseline types ({len(non_media_baseline_values_filled)})."
    )

  baseline_list = []
  for channel in range(non_media_treatments.shape[-1]):
    baseline_value = non_media_baseline_values_filled[channel]

    if baseline_value == constants.NON_MEDIA_BASELINE_MIN:
      baseline_for_channel = tf.reduce_min(
          non_media_treatments[..., channel], axis=[0, 1]
      )
    elif baseline_value == constants.NON_MEDIA_BASELINE_MAX:
      baseline_for_channel = tf.reduce_max(
          non_media_treatments[..., channel], axis=[0, 1]
      )
    elif isinstance(baseline_value, float):
      baseline_for_channel = tf.cast(baseline_value, tf.float32)
    else:
      raise ValueError(
          f"Invalid non_media_baseline_values value: '{baseline_value}'. Only"
          " float numbers and strings 'min' and 'max' are supported."
      )

    baseline_list.append(
        baseline_for_channel
        * tf.ones_like(non_media_treatments[..., channel])
        * non_media_selected_times
    )

  return tf.stack(baseline_list, axis=-1)


class Analyzer:
  """Runs calculations to analyze the raw data after fitting the model."""

  def __init__(self, meridian: model.Meridian):
    self._meridian = meridian
    # Make the meridian object ready for methods in this analyzer that create
    # tf.function computation graphs: it should be frozen for no more internal
    # states mutation before those graphs execute.
    self._meridian.populate_cached_properties()

  @tf.function(jit_compile=True)
  def _get_kpi_means(
      self,
      data_tensors: DataTensors,
      dist_tensors: DistributionTensors,
  ) -> Tensor:
    """Computes batched KPI means.

    Note that the output array has the same number of time periods as the media
    data (lagged time periods are included).

    Args:
      data_tensors: A `DataTensors` container with the following tensors:
        `media`, `reach`, `frequency`, `organic_media`, `organic_reach`,
        `organic_frequency`, `non_media_treatments`, `controls`. The `media`,
        `reach`, `organic_media`, `organic_reach` and `non_media_treatments`
        tensors are assumed to be scaled by their corresponding transformers.
      dist_tensors: A `DistributionTensors` container with the distribution
        tensors for media, RF, organic media, organic RF, non-media treatments,
        and controls.

    Returns:
      Tensor representing computed kpi means.
    """
    tau_gt = expand_dims(dist_tensors.tau_g, -1) + expand_dims(
        dist_tensors.mu_t, -2
    )
    combined_media_transformed, combined_beta = (
        self._get_transformed_media_and_beta(
            data_tensors=data_tensors,
            dist_tensors=dist_tensors,
        )
    )

    result = (
        tau_gt
        + einsum(
            "...gtm,...gm->...gt", combined_media_transformed, combined_beta
        )
        + einsum(
            "...gtc,...gc->...gt",
            data_tensors.controls,
            dist_tensors.gamma_gc,
        )
    )
    if data_tensors.non_media_treatments is not None:
      result += einsum(
          "...gtm,...gm->...gt",
          data_tensors.non_media_treatments,
          dist_tensors.gamma_gn,
      )
    return result

  def _check_revenue_data_exists(self, use_kpi: bool = False):
    """Checks if the revenue data is available for the analysis.

    In the `kpi_type=NON_REVENUE` case, `revenue_per_kpi` is required to perform
    the revenue analysis. If `revenue_per_kpi` is not defined, then the revenue
    data is not available and the revenue analysis (`use_kpi=False`) is not
    possible. Only the KPI analysis (`use_kpi=True`) is possible in this case.

    In the `kpi_type=REVENUE` case, KPI is equal to revenue and setting
    `use_kpi=True` has no effect. Therefore, a warning is issued if the default
    `False` value of `use_kpi` is overridden by the user.

    Args:
      use_kpi: A boolean flag indicating whether to use KPI instead of revenue.

    Raises:
      ValueError: If `use_kpi` is `False` and `revenue_per_kpi` is not defined.
      UserWarning: If `use_kpi` is `True` in the `kpi_type=REVENUE` case.
    """
    if self._meridian.input_data.kpi_type == constants.NON_REVENUE:
      if not use_kpi and self._meridian.revenue_per_kpi is None:
        raise ValueError(
            "Revenue analysis is not available when `revenue_per_kpi` is"
            " unknown. Set `use_kpi=True` to perform KPI analysis instead."
        )

    if self._meridian.input_data.kpi_type == constants.REVENUE:
      # In the `kpi_type=REVENUE` case, KPI is equal to revenue and
      # `revenue_per_kpi` is set to a tensor of 1s in the initialization of the
      # `InputData` object.
      assert self._meridian.revenue_per_kpi is not None
      if use_kpi:
        warnings.warn(
            "Setting `use_kpi=True` has no effect when `kpi_type=REVENUE`"
            " since in this case, KPI is equal to revenue."
        )

  def _get_adstock_dataframe(
      self,
      channel_type: str,
      l_range: np.ndarray,
      xr_dims: Sequence[str],
      xr_coords: Mapping[str, tuple[Sequence[str], Sequence[str]]],
      confidence_level: float = constants.DEFAULT_CONFIDENCE_LEVEL,
  ) -> pd.DataFrame:
    """Computes decayed effect means and CIs for media or RF channels.

    Args:
      channel_type: Specifies `media`, `reach`, or `organic_media` for computing
        prior and posterior decayed effects.
      l_range: The range of time across which the adstock effect is computed.
      xr_dims: A list of dimensions for the output dataset.
      xr_coords: A dictionary with the coordinates for the output dataset.
      confidence_level: Confidence level for computing credible intervals,
        represented as a value between zero and one.

    Returns:
      Pandas DataFrame containing the channel, time_units, distribution, ci_hi,
      ci_lo, and mean decayed effects for either media or RF channel types.
    """
    if channel_type is constants.MEDIA:
      prior = self._meridian.inference_data.prior.alpha_m.values[0]
      posterior = np.reshape(
          self._meridian.inference_data.posterior.alpha_m.values,
          (-1, self._meridian.n_media_channels),
      )
    elif channel_type is constants.REACH:
      prior = self._meridian.inference_data.prior.alpha_rf.values[0]
      posterior = np.reshape(
          self._meridian.inference_data.posterior.alpha_rf.values,
          (-1, self._meridian.n_rf_channels),
      )
    elif channel_type is constants.ORGANIC_MEDIA:
      prior = self._meridian.inference_data.prior.alpha_om.values[0]
      posterior = np.reshape(
          self._meridian.inference_data.posterior.alpha_om.values,
          (-1, self._meridian.n_organic_media_channels),
      )
    else:
      raise ValueError(
          f"Unsupported channel type for adstock decay: '{channel_type}'. "
      )

    decayed_effect_prior = (
        prior[np.newaxis, ...] ** l_range[:, np.newaxis, np.newaxis, np.newaxis]
    )
    decayed_effect_posterior = (
        posterior[np.newaxis, ...]
        ** l_range[:, np.newaxis, np.newaxis, np.newaxis]
    )

    decayed_effect_prior_transpose = transpose(
        decayed_effect_prior, perm=[1, 2, 0, 3]
    )
    decayed_effect_posterior_transpose = transpose(
        decayed_effect_posterior, perm=[1, 2, 0, 3]
    )
    adstock_dataset = _central_tendency_and_ci_by_prior_and_posterior(
        decayed_effect_prior_transpose,
        decayed_effect_posterior_transpose,
        constants.EFFECT,
        xr_dims,
        xr_coords,
        confidence_level,
    )
    return (
        adstock_dataset[constants.EFFECT]
        .to_dataframe()
        .reset_index()
        .pivot(
            index=[
                constants.CHANNEL,
                constants.TIME_UNITS,
                constants.DISTRIBUTION,
            ],
            columns=constants.METRIC,
            values=constants.EFFECT,
        )
        .reset_index()
    )

  def _get_scaled_data_tensors(
      self,
      new_data: DataTensors | None = None,
      include_non_paid_channels: bool = True,
  ) -> DataTensors:
    """Get scaled tensors using given new data and original data.

    This method returns a new `DataTensors` container with scaled versions of
    `media`, `reach`, `frequency`, `organic_media`, `organic_reach`,
    `organic_frequency`, `non_media_treatments`, `controls` and
    `revenue_per_kpi` tensors. For each tensor, if its value is provided in the
    `new_data` argument, the provided tensors are used. Otherwise the original
    tensors from the Meridian model are used. The tensors are then either scaled
    by their corresponding transformers (`media`, `reach`, `organic_media`,
    `organic_reach`, `non_media_treatments`, `controls`), or left as is
    (`frequency`, `organic_frequency`, `revenue_per_kpi`). For example,

    ```
    _get_scaled_data_tensors(
        new_data=DataTensors(media=new_media),
    )
    ```

    returns a `DataTensors` container with `media` set to the scaled version of
    `new_media`, and all other tensors set to their original scaled values from
    the Meridian model.

    Args:
      new_data: An optional `DataTensors` container with optional tensors:
        `media`, `reach`, `frequency`, `organic_media`, `organic_reach`,
        `organic_frequency`, `non_media_treatments`, `controls` and
        `revenue_per_kpi`. If `None`, the original scaled tensors from the
        Meridian object are used. If `new_data` is provided, the output contains
        the scaled versions of the tensors in `new_data` and the original scaled
        versions of all the remaining tensors. The new tensors' dimensions must
        match the dimensions of the corresponding original tensors from
        `meridian.input_data`.
      include_non_paid_channels: Boolean. If `True`, organic media, organic RF
        and non-media treatments data is included in the output.

    Returns:
      A DataTensors object containing the scaled `media`, `reach`, `frequency`
      `organic_media`, `organic_reach`, `organic_frequency`,
      `non_media_treatments`, `controls` and `revenue_per_kpi` data tensors.
    """
    if new_data is None:
      return DataTensors(
          media=self._meridian.media_tensors.media_scaled,
          reach=self._meridian.rf_tensors.reach_scaled,
          frequency=self._meridian.rf_tensors.frequency,
          organic_media=self._meridian.organic_media_tensors.organic_media_scaled,
          organic_reach=self._meridian.organic_rf_tensors.organic_reach_scaled,
          organic_frequency=self._meridian.organic_rf_tensors.organic_frequency,
          non_media_treatments=self._meridian.non_media_treatments_scaled,
          controls=self._meridian.controls_scaled,
          revenue_per_kpi=self._meridian.revenue_per_kpi,
      )
    media_scaled = _transformed_new_or_scaled(
        new_variable=new_data.media,
        transformer=self._meridian.media_tensors.media_transformer,
        scaled_variable=self._meridian.media_tensors.media_scaled,
    )

    reach_scaled = _transformed_new_or_scaled(
        new_variable=new_data.reach,
        transformer=self._meridian.rf_tensors.reach_transformer,
        scaled_variable=self._meridian.rf_tensors.reach_scaled,
    )

    frequency = (
        new_data.frequency
        if new_data.frequency is not None
        else self._meridian.rf_tensors.frequency
    )

    controls_scaled = _transformed_new_or_scaled(
        new_variable=new_data.controls,
        transformer=self._meridian.controls_transformer,
        scaled_variable=self._meridian.controls_scaled,
    )
    revenue_per_kpi = (
        new_data.revenue_per_kpi
        if new_data.revenue_per_kpi is not None
        else self._meridian.revenue_per_kpi
    )

    if include_non_paid_channels:
      organic_media_scaled = _transformed_new_or_scaled(
          new_variable=new_data.organic_media,
          transformer=self._meridian.organic_media_tensors.organic_media_transformer,
          scaled_variable=self._meridian.organic_media_tensors.organic_media_scaled,
      )
      organic_reach_scaled = _transformed_new_or_scaled(
          new_variable=new_data.organic_reach,
          transformer=self._meridian.organic_rf_tensors.organic_reach_transformer,
          scaled_variable=self._meridian.organic_rf_tensors.organic_reach_scaled,
      )
      organic_frequency = (
          new_data.organic_frequency
          if new_data.organic_frequency is not None
          else self._meridian.organic_rf_tensors.organic_frequency
      )
      non_media_treatments_scaled = _transformed_new_or_scaled(
          new_variable=new_data.non_media_treatments,
          transformer=self._meridian.non_media_transformer,
          scaled_variable=self._meridian.non_media_treatments_scaled,
      )
      return DataTensors(
          media=media_scaled,
          reach=reach_scaled,
          frequency=frequency,
          organic_media=organic_media_scaled,
          organic_reach=organic_reach_scaled,
          organic_frequency=organic_frequency,
          non_media_treatments=non_media_treatments_scaled,
          controls=controls_scaled,
          revenue_per_kpi=revenue_per_kpi,
      )
    else:
      return DataTensors(
          media=media_scaled,
          reach=reach_scaled,
          frequency=frequency,
          controls=controls_scaled,
          revenue_per_kpi=revenue_per_kpi,
      )

  def _get_causal_param_names(
      self,
      include_non_paid_channels: bool,
  ) -> list[str]:
    """Gets media, RF, non-media, organic media, and organic RF distributions.

    Args:
      include_non_paid_channels: Boolean. If `True`, organic media, organic RF
        and non-media treatments data is included in the output.

    Returns:
      A list containing available media, RF, non-media treatments, organic media
      and organic RF parameters names in inference data.
    """
    params = []
    if self._meridian.media_tensors.media is not None:
      params.extend([
          constants.EC_M,
          constants.SLOPE_M,
          constants.ALPHA_M,
          constants.BETA_GM,
      ])
    if self._meridian.rf_tensors.reach is not None:
      params.extend([
          constants.EC_RF,
          constants.SLOPE_RF,
          constants.ALPHA_RF,
          constants.BETA_GRF,
      ])
    if include_non_paid_channels:
      if self._meridian.organic_media_tensors.organic_media is not None:
        params.extend([
            constants.EC_OM,
            constants.SLOPE_OM,
            constants.ALPHA_OM,
            constants.BETA_GOM,
        ])
      if self._meridian.organic_rf_tensors.organic_reach is not None:
        params.extend([
            constants.EC_ORF,
            constants.SLOPE_ORF,
            constants.ALPHA_ORF,
            constants.BETA_GORF,
        ])
      if self._meridian.non_media_treatments is not None:
        params.extend([
            constants.GAMMA_GN,
        ])
    return params

  def _get_transformed_media_and_beta(
      self,
      data_tensors: DataTensors,
      dist_tensors: DistributionTensors,
      n_times_output: int | None = None,
  ) -> tuple[Tensor | None, Tensor | None]:
    """Function for transforming media using adstock and hill functions.

    This transforms the media tensor using the adstock and hill functions, in
    the desired order.

    Args:
      data_tensors: A `DataTensors` container with the following tensors:
        `media`, `reach`, `frequency`, `organic_media`, `organic_reach`,
        `organic_frequency`.
      dist_tensors: A `DistributionTensors` container with the distribution
        tensors for media, RF, organic media, and organic RF channels.
      n_times_output: Optional number of time periods to output. Defaults to the
        corresponding argument defaults for `adstock_hill_media` and
        `adstock_hill_rf`.

    Returns:
      A tuple `(combined_media_transformed, combined_beta)`.
    """
    combined_medias = []
    combined_betas = []
    if data_tensors.media is not None:
      combined_medias.append(
          self._meridian.adstock_hill_media(
              media=data_tensors.media,
              alpha=dist_tensors.alpha_m,
              ec=dist_tensors.ec_m,
              slope=dist_tensors.slope_m,
              n_times_output=n_times_output,
          )
      )
      combined_betas.append(dist_tensors.beta_gm)

    if data_tensors.reach is not None:
      combined_medias.append(
          self._meridian.adstock_hill_rf(
              reach=data_tensors.reach,
              frequency=data_tensors.frequency,
              alpha=dist_tensors.alpha_rf,
              ec=dist_tensors.ec_rf,
              slope=dist_tensors.slope_rf,
              n_times_output=n_times_output,
          )
      )
      combined_betas.append(dist_tensors.beta_grf)
    if data_tensors.organic_media is not None:
      combined_medias.append(
          self._meridian.adstock_hill_media(
              media=data_tensors.organic_media,
              alpha=dist_tensors.alpha_om,
              ec=dist_tensors.ec_om,
              slope=dist_tensors.slope_om,
              n_times_output=n_times_output,
          )
      )
      combined_betas.append(dist_tensors.beta_gom)
    if data_tensors.organic_reach is not None:
      combined_medias.append(
          self._meridian.adstock_hill_rf(
              reach=data_tensors.organic_reach,
              frequency=data_tensors.organic_frequency,
              alpha=dist_tensors.alpha_orf,
              ec=dist_tensors.ec_orf,
              slope=dist_tensors.slope_orf,
              n_times_output=n_times_output,
          )
      )
      combined_betas.append(dist_tensors.beta_gorf)

    combined_media_transformed = concat(combined_medias, axis=-1)
    combined_beta = concat(combined_betas, axis=-1)
    return combined_media_transformed, combined_beta
