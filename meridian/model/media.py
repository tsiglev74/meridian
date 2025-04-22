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

"""Structures and functions for manipulating media value data and tensors."""

import dataclasses
from collections.abc import Mapping
from typing import Optional
from meridian import constants
from meridian.backend import (
    Tensor,
    cast,
    convert_to_tensor,
    experimental_extension_type,
    zeros_like,
)
from meridian.data import input_data as data
from meridian.model import spec
from meridian.model import transformers


__all__ = [
    "MediaTensors",
    "RfTensors",
    "OrganicMediaTensors",
    "OrganicRfTensors",
    "build_media_tensors",
    "build_rf_tensors",
    "build_organic_media_tensors",
    "build_organic_rf_tensors",
]


@dataclasses.dataclass(frozen=True)
class MediaTensors:
  """Container for media tensors.

  Attributes:
    media: A tensor constructed from `InputData.media`.
    media_spend: A tensor constructed from `InputData.media_spend`.
    media_transformer: A `MediaTransformer` to scale media tensors using the
      model's media data.
    media_scaled: The media tensor normalized by population and by the median
      value.
    media_counterfactual: A tensor containing the media counterfactual values.
      If ROI priors are used, then the ROI of media channels is based on the
      difference in expected sales between the `media` tensor and this
      `media_counterfactual` tensor.
    media_counterfactual_scaled: A tensor containing the media counterfactual
      scaled values.
    media_spend_counterfactual: A tensor containing the media spend
      counterfactual values. If ROI priors are used, then the ROI of media
      channels is based on the spend difference between `media_spend` tensor and
      this `media_spend_counterfactual` tensor.
  """

  media: Tensor | None = None
  media_spend: Tensor | None = None
  media_transformer: transformers.MediaTransformer | None = None
  media_scaled: Tensor | None = None
  media_counterfactual: Tensor | None = None
  media_counterfactual_scaled: Tensor | None = None
  media_spend_counterfactual: Tensor | None = None


def build_media_tensors(
    input_data: data.InputData,
    model_spec: spec.ModelSpec,
) -> MediaTensors:
  """Derives a MediaTensors container from media values in given input data."""
  if input_data.media is None:
    return MediaTensors()

  # Derive and set media tensors from media values in the input data.
  media = convert_to_tensor(input_data.media, dtype=float)
  media_spend = convert_to_tensor(input_data.media_spend, dtype=float)
  media_transformer = transformers.MediaTransformer(
      media, convert_to_tensor(input_data.population, dtype=float)
  )
  media_scaled = media_transformer.forward(media)

  # Derive counterfactual media tensors depending on whether mroi or roi priors
  # are used and whether roi_calibration_period is specified.
  if (
      model_spec.media_prior_type
      == constants.TREATMENT_PRIOR_TYPE_MROI
  ):
    factor = constants.MROI_FACTOR
  else:
    factor = 0

  if model_spec.roi_calibration_period is None:
    media_counterfactual = factor * media
    media_counterfactual_scaled = factor * media_scaled
    media_spend_counterfactual = factor * media_spend
  else:
    media_counterfactual = cast(
        media,
        model_spec.roi_calibration_period,
        factor * media,
        media,
    )
    media_counterfactual_scaled = cast(
        media_scaled,
        model_spec.roi_calibration_period,
        factor * media_scaled,
        media_scaled,
    )
    n_times = len(input_data.time)
    media_spend_counterfactual = cast(
        media_spend,
        model_spec.roi_calibration_period[..., -n_times:, :],
        factor * media_spend,
        media_spend,
    )

  return MediaTensors(
      media=media,
      media_spend=media_spend,
      media_transformer=media_transformer,
      media_scaled=media_scaled,
      media_counterfactual=media_counterfactual,
      media_counterfactual_scaled=media_counterfactual_scaled,
      media_spend_counterfactual=media_spend_counterfactual,
  )


@dataclasses.dataclass(frozen=True)
class OrganicMediaTensors:
  """Container for organic media tensors.

  Attributes:
    organic_media: A tensor constructed from `InputData.organic_media`.
    organic_media_transformer: A `MediaTransformer` to scale media tensors using
      the model's organic media data.
    organic_media_scaled: The organic media tensor normalized by population and
      by the median value.
    organic_media_counterfactual: A tensor containing the organic media
      counterfactual values.
    organic_media_counterfactual_scaled: A tensor containing the organic media
      counterfactual scaled values.
  """

  organic_media: Tensor | None = None
  organic_media_transformer: transformers.MediaTransformer | None = None
  organic_media_scaled: Tensor | None = None
  organic_media_counterfactual: Tensor | None = None
  organic_media_counterfactual_scaled: Tensor | None = None


def build_organic_media_tensors(
    input_data: data.InputData,
) -> OrganicMediaTensors:
  """Derives a OrganicMediaTensors container from values in given input data."""
  if input_data.organic_media is None:
    return OrganicMediaTensors()

  # Derive and set media tensors from media values in the input data.
  organic_media = convert_to_tensor(
      input_data.organic_media, dtype=float
  )
  organic_media_transformer = transformers.MediaTransformer(
      organic_media,
      convert_to_tensor(input_data.population, dtype=float),
  )
  organic_media_scaled = organic_media_transformer.forward(organic_media)
  organic_media_counterfactual = zeros_like(organic_media)
  organic_media_counterfactual_scaled = zeros_like(organic_media_scaled)

  return OrganicMediaTensors(
      organic_media=organic_media,
      organic_media_transformer=organic_media_transformer,
      organic_media_scaled=organic_media_scaled,
      organic_media_counterfactual=organic_media_counterfactual,
      organic_media_counterfactual_scaled=organic_media_counterfactual_scaled,
  )


@dataclasses.dataclass(frozen=True)
class RfTensors:
  """Container for Reach and Frequency (RF) media tensors.

  Attributes:
    reach: A tensor constructed from `InputData.reach`.
    frequency: A tensor constructed from `InputData.frequency`.
    rf_spend: A tensor constructed from `InputData.rf_spend`.
    reach_transformer: A `MediaTransformer` to scale RF tensors using the
      model's RF data.
    reach_scaled: A reach tensor normalized by population and by the median
      value.
    reach_counterfactual: A reach tensor with media counterfactual values. If
      ROI priors are used, then the ROI of R&F channels is based on the
      difference in expected sales between the `reach` tensor and this
      `reach_counterfactual` tensor.
    reach_counterfactual_scaled: A reach tensor with media counterfactual scaled
      values.
    rf_spend_counterfactual: A reach tensor with media spend counterfactual
      values. If ROI priors are used, then the ROI of R&F channels is based on
      the spend difference between `rf_spend` tensor and this
      `rf_spend_counterfactual` tensor.
  """

  reach: Tensor | None = None
  frequency: Tensor | None = None
  rf_spend: Tensor | None = None
  reach_transformer: transformers.MediaTransformer | None = None
  reach_scaled: Tensor | None = None
  reach_counterfactual: Tensor | None = None
  reach_counterfactual_scaled: Tensor | None = None
  rf_spend_counterfactual: Tensor | None = None


def build_rf_tensors(
    input_data: data.InputData,
    model_spec: spec.ModelSpec,
) -> RfTensors:
  """Derives an RfTensors container from RF media values in given input."""
  if input_data.reach is None:
    return RfTensors()

  reach = convert_to_tensor(input_data.reach, dtype=float)
  frequency = convert_to_tensor(input_data.frequency, dtype=float)
  rf_spend = convert_to_tensor(input_data.rf_spend, dtype=float)
  reach_transformer = transformers.MediaTransformer(
      reach, convert_to_tensor(input_data.population, dtype=float)
  )
  reach_scaled = reach_transformer.forward(reach)

  # marginal ROI by reach equals the ROI. The conversion between `beta_rf` and
  # `roi_rf` is the same, regardless of whether `roi_rf` represents ROI or
  # marginal ROI by reach.
  if model_spec.rf_roi_calibration_period is None:
    reach_counterfactual = zeros_like(reach)
    reach_counterfactual_scaled = zeros_like(reach_scaled)
    rf_spend_counterfactual = zeros_like(rf_spend)
  else:
    reach_counterfactual = cast(
        reach,
        model_spec.rf_roi_calibration_period,
        0,
        reach,
    )
    reach_counterfactual_scaled = cast(
        reach_scaled,
        model_spec.rf_roi_calibration_period,
        0,
        reach_scaled,
    )
    n_times = len(input_data.time)
    rf_spend_counterfactual = cast(
        rf_spend,
        model_spec.rf_roi_calibration_period[..., -n_times:, :],
        0,
        rf_spend,
    )

  return RfTensors(
      reach=reach,
      frequency=frequency,
      rf_spend=rf_spend,
      reach_transformer=reach_transformer,
      reach_scaled=reach_scaled,
      reach_counterfactual=reach_counterfactual,
      reach_counterfactual_scaled=reach_counterfactual_scaled,
      rf_spend_counterfactual=rf_spend_counterfactual,
  )


@dataclasses.dataclass(frozen=True)
class OrganicRfTensors:
  """Container for Reach and Frequency (RF) organic media tensors.

  Attributes:
    organic_reach: A tensor constructed from `InputData.organic_reach`.
    organic_frequency: A tensor constructed from `InputData.organic_frequency`.
    organic_reach_transformer: A `MediaTransformer` to scale organic RF tensors
      using the model's organic RF data.
    organic_reach_scaled: An organic reach tensor normalized by population and
      by the median value.
    organic_reach_counterfactual: An organic reach tensor with media
      counterfactual values.
    organic_reach_counterfactual_scaled: An organic reach tensor with media
      counterfactual scaled values.
  """

  organic_reach: Tensor | None = None
  organic_frequency: Tensor | None = None
  organic_reach_transformer: transformers.MediaTransformer | None = None
  organic_reach_scaled: Tensor | None = None
  organic_reach_counterfactual: Tensor | None = None
  organic_reach_counterfactual_scaled: Tensor | None = None


def build_organic_rf_tensors(
    input_data: data.InputData,
) -> OrganicRfTensors:
  """Derives an OrganicRfTensors container from values in given input."""
  if input_data.organic_reach is None:
    return OrganicRfTensors()

  organic_reach = convert_to_tensor(
      input_data.organic_reach, dtype=float
  )
  organic_frequency = convert_to_tensor(
      input_data.organic_frequency, dtype=float
  )
  organic_reach_transformer = transformers.MediaTransformer(
      organic_reach,
      convert_to_tensor(input_data.population, dtype=float),
  )
  organic_reach_scaled = organic_reach_transformer.forward(organic_reach)
  organic_reach_counterfactual = zeros_like(organic_reach)
  organic_reach_counterfactual_scaled = zeros_like(organic_reach_scaled)

  return OrganicRfTensors(
      organic_reach=organic_reach,
      organic_frequency=organic_frequency,
      organic_reach_transformer=organic_reach_transformer,
      organic_reach_scaled=organic_reach_scaled,
      organic_reach_counterfactual=organic_reach_counterfactual,
      organic_reach_counterfactual_scaled=organic_reach_counterfactual_scaled,
  )
