"""Contains classes and methods to train a Meridian model.

The DataLoaderParams dataclass validates and holds the parameters required to
create an InputData object, and provides a cached data() property to load data
from the data_path field.
"""

import abc
import dataclasses
import functools
import os
import time
from typing import Any, Mapping, Sequence
import warnings

from absl import app
from absl import flags
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import xarray as xr

from google3.ads.lift.mmm.modeling.distribute import utils
from google3.ads.lift.mmm.modeling.research.contr_priors import constants as mc
from google3.ads.lift.mmm.modeling.research.contr_priors import train_constants as c
import google3.ads.lift.mmm.modeling.research.contr_priors.analysis.optimizer as opt
from google3.ads.lift.mmm.modeling.research.contr_priors.data import input_data
from google3.ads.lift.mmm.modeling.research.contr_priors.data import load
from google3.ads.lift.mmm.modeling.research.contr_priors.model import model
from google3.ads.lift.mmm.modeling.research.contr_priors.model import prior_distribution
from google3.ads.lift.mmm.modeling.research.contr_priors.model import spec
from google3.pyglib import gfile
from google3.pyglib.contrib.gpathlib import gpath


_LDAP = flags.DEFINE_string(
    "ldap",
    None,
    "Ldap for config directory.",
    required=True,
)
_CONFIG = flags.DEFINE_string(
    "config_path", None, "Config file path for experiment run.", required=True
)
_PRIOR_MAPPING_TYPES = (
    Mapping[str, str | float | Mapping[str, str | float]] | None
)

_USE_GPU = os.environ.get(c.USE_GPU)
_N_GPUS = os.environ.get(c.N_GPUS)
_N_CPUS = os.environ.get(c.N_CPUS)
_RAM_GIB = os.environ.get(c.RAM_GIB)


def get_channel_mapping(
    columns: Sequence[str] | None,
) -> Mapping[str, str] | None:
  """Get channel name mapping from all columns.

  Args:
    columns: Names of columns in a pandas DataFrame.

  Returns:
    Mapping of columns to channel names embedded in `columns.`
  """
  return (
      {column: get_channel_name_from_column(column) for column in columns}
      if columns
      else None
  )


def get_channel_name_from_column(column: str) -> str:
  """Return channel name from full column string.

  Args:
    column: Name of a single column from a pandas DataFrame.

  Returns:
    channel name embedding in `column.`
  """
  return "_".join(column.split("_")[1:])


def _validate_path(path_name: str, path: str):
  """Validates the path of a data path parameter.

  Args:
    path_name: Name of which path is being validated. Used for clear exception
      messaging.
    path: Path of a data file to load.
  """
  if not gfile.Exists(path):
    raise ValueError(f"{path_name} {path} does not exist.")


def _validate_type(
    param_name: str, param_value: Any, allowed_types: Any | Sequence[Any]
):
  """Validates the type of the parameter."""
  if not isinstance(param_value, allowed_types):
    raise ValueError(
        f"{param_name} must be in {allowed_types}, got" + f" {param_value}."
    )


def _unpack_bijector_params(params):
  """Returns a tfp.bijectors.Bijector object based on configuration parameters.

  Args:
    params: Parameters for tfp.distribution.Bijector creation.

  Returns:
    tfp.bijectors.Bijector from specified configuration parameters.
  """
  bijector_type = getattr(tfp.bijectors, params.pop(c.BIJECTOR_TYPE))
  return bijector_type(**params)


def _unpack_distribution_params(params):
  """Parses a mapping of priors to distribution parameters.

  Args:
    params: Parameters for tfp.distribution.Distribution creation.

  Returns:
    prior attribute to prior parameter mapping.
  """
  if c.DISTRIBUTION in params:
    params[c.DISTRIBUTION] = _unpack_distribution_params(params[c.DISTRIBUTION])
  if c.BIJECTOR in params:
    params[c.BIJECTOR] = _unpack_bijector_params(params[c.BIJECTOR])
  params[c.DISTRIBUTION_TYPE] = getattr(
      tfp.distributions, params[c.DISTRIBUTION_TYPE]
  )
  return params


def _create_prior_distribution(
    custom_priors,
) -> prior_distribution.PriorDistribution:
  """Creates a PriorDistribution object from the specified custom priors.

  Example of the custom_priors argument:

  ```python
  {
    "roi_m": {
      "loc": 0.4,
      "scale": 0.4,
      "distribution_type":"LogNormal",
    },
    "ec_m": {
      "distribution": {
        "scale": 0.5,
        "distribution_type": "HalfNormal",
      },
      "bijector": {
        "shift": 0.1,
        "bijector_type": "Shift",
      },
      "distribution_type": "TransformedDistribution",
    },
  }
  ```

  As the example shows, complex distribution types such as
  `tfp.distributions.TransformedDistribution`, can be configured using
  nested maps.

  Args:
    custom_priors: Mapping of distribution parameters. Each top level key should
      be the string name of a PriorDistribution attribute. Each key should have
      at a minimum a distribution_type string, which will be converted to the
      equivalent tfp.distributions type. The remaining key value pairs will
      depend on the type of distribution used.

  Returns:
    PriorDistribution for `spec.ModelSpec` object used for `meridian.model`
      model fitting.
  """
  custom_dict = {}
  for prior_attr, prior_dict in custom_priors.items():
    custom_dict[prior_attr] = _unpack_distribution_params(prior_dict)
    custom_dict[prior_attr][c.NAME] = prior_attr
  prior = prior_distribution.PriorDistribution()
  prior.__setstate__(custom_dict)
  return prior


@dataclasses.dataclass
class DataLoaderParams(metaclass=abc.ABCMeta):
  """Validates and holds data loader parameters from the specified format.

  Attributes:
    data: Dataset containing controls, media, media_spend, reach, frequency,
      rf_spend, population, kpi, and reveneue per kpi data.
    data_path: Path to the dataset to be loaded.
    kpi_type: A string denoting whether the kpi is of a `revenue` or `non-
      revenue` type. When the `kpi_type` is `non-revenue` and there exists a
      `revenue_per_kpi`, we use ROI calibration and the analysis is run on
      `revenue`, and when the revenue_per_kpi doesn't exist for the same
      `kpi_type`, we use custom ROI calibration and the analysis is run on KPI.
    path_type: Type of data to be loaded. Currently supports `xarray.`
  """

  data_path: str
  kpi_type: str
  path_type: str

  def __post_init__(self):
    self._validate_data_args()

  @functools.cached_property
  def data(self) -> xr.Dataset | pd.DataFrame:
    return self._data()

  @abc.abstractmethod
  def _data(self):
    """Reads the data and outputs an InputData object."""
    raise NotImplementedError()

  def _validate_data_args(self):
    """Validates data arguments for DataLoader."""
    _validate_type(c.KPI_TYPE, self.kpi_type, str)
    _validate_type(c.DATA_PATH, self.data_path, str)
    _validate_type(c.PATH_TYPE, self.path_type, str)
    _validate_path(c.DATA_PATH, self.data_path)


@dataclasses.dataclass
class XarrayDataLoaderParams(DataLoaderParams):
  """Validates and holds data loader parameters.

  Attributes:
    data: Dataset containing controls, media, media_spend, reach, frequency,
      rf_spend, population, kpi, and reveneue per kpi data.
    data_path: Path to the dataset to be loaded.
    kpi_type: A string denoting whether the kpi is of a `revenue` or `non-
      revenue` type. When the `kpi_type` is `non-revenue` and there exists a
      `revenue_per_kpi`, we use ROI calibration and the analysis is run on
      `revenue`, and when the revenue_per_kpi doesn't exist for the same
      `kpi_type`, we use custom ROI calibration and the analysis is run on KPI.
    path_type: Type of data to be loaded. Currently supports `xarray.`
  """

  data_path: str
  kpi_type: str
  path_type: str

  def _data(self) -> xr.Dataset:
    """Reads the data from the specified data path.

    Returns:
      xarray.Dataset containing controls, media, media_spend, reach,
        frequency, rf_spend, organic_media, organic_reach, organic_frequency,
        non_media_treatments, population, kpi, and reveneue per kpi data. For
        more details on expected data format, see:
        https://source.corp.google.com/piper///depot/google3/third_party/py/meridian/data/load.py.
    """
    with gfile.Open(self.data_path, "rb") as f:
      return joblib.load(f)


@dataclasses.dataclass
class DataFrameDataLoaderParams(DataLoaderParams):
  """Validates and holds data loader parameters.

  Attributes:
    coord_to_columns: A CoordToColumns object for mapping dataframe column names
      to InputData columns.
    data: A dataset containing controls, media, media_spend, reach, frequency,
      rf_spend, population, kpi, and reveneue per kpi data.
    data_path: Path to the dataset to be loaded.
    kpi_type: A string denoting whether the kpi is of a `revenue` or `non-
      revenue` type. When the `kpi_type` is `non-revenue` and there exists a
      `revenue_per_kpi`, we use ROI calibration and the analysis is run on
      `revenue`, and when the revenue_per_kpi doesn't exist for the same
      `kpi_type`, we use custom ROI calibration and the analysis is run on KPI.
    path_type: Type of data to be loaded. Currently supports  pandas
      `DataFrame.`
    geo_column: A column in `data` containing geo labels.
    time_column: A column in `data` containing time labels.
    kpi_column: A column in `data` containing kpi data.
    n_dma_to_keep: Number of DMAs to keep in the data.
    n_lag_history_periods: Number of time periods to throw away from non-media
      columns.
    include_gqv: Whether to include gqv labeled columns in `InputData.controls`.
    revenue_per_kpi_column: A column in `data` contining revenue_per_kpi data.
    media_to_channel: A dictionary whose keys are the actual column names for
      media data in the dataframe and values are the desired channel names, the
      same as for the media_spend data. Example: ``` media_to_channel =
      {'media_tv': 'tv', 'media_yt': 'yt', 'media_fb': 'fb'} ```
    media_spend_to_channel: A dictionary whose keys are the actual column names
      for media_spend data in the dataframe and values are the desired channel
      names, the same as for the media data. Example: ``` media_spend_to_channel
      = {'spend_tv': 'tv', 'spend_yt': 'yt', 'spend_fb': 'fb'} ```
    reach_to_channel: A dictionary whose keys are the actual column names for
      reach data in the dataframe and values are the desired channel names, the
      same as for the RF spend data. Example: ``` reach_to_channel =
      {'reach_tv': 'tv', 'reach_yt': 'yt', 'reach_fb': 'fb'} ```
    frequency_to_channel: A dictionary whose keys are the actual column names
      for frequency data in the dataframe and values are the desired channel
      names, the same as for the RF spend data. Example: ```
      frequency_to_channel = {'frequency_tv': 'tv', 'frequency_yt': 'yt',
      'frequency_fb': 'fb'} ```
    rf_spend_to_channel: A dictionary whose keys are the actual column names for
      RF spend data in the dataframe and values are the desired channel names,
      the same as for the reach and frequency data. Example: ```
      rf_spend_to_channel = {'rf_spend_tv': 'tv', 'rf_spend_yt': 'yt',
      'rf_spend_fb': 'fb'} ```
  """

  data_path: str
  path_type: str
  kpi_type: str
  geo_column: str
  time_column: str
  kpi_column: str
  n_dma_to_keep: int
  n_lag_history_periods: int
  include_gqv: bool
  revenue_per_kpi_column: str | None = None

  @functools.cached_property
  def coord_to_columns(self) -> load.CoordToColumns:
    return self._get_coord_to_columns()

  @functools.cached_property
  def frequency_to_channel(self) -> Mapping[str, str] | None:
    return get_channel_mapping(self.coord_to_columns.frequency)

  @functools.cached_property
  def media_to_channel(self) -> Mapping[str, str] | None:
    return get_channel_mapping(self.coord_to_columns.media)

  @functools.cached_property
  def media_spend_to_channel(self) -> Mapping[str, str] | None:
    return get_channel_mapping(self.coord_to_columns.media_spend)

  @functools.cached_property
  def reach_to_channel(self) -> Mapping[str, str] | None:
    return get_channel_mapping(self.coord_to_columns.reach)

  @functools.cached_property
  def rf_spend_to_channel(self) -> Mapping[str, str] | None:
    return get_channel_mapping(self.coord_to_columns.rf_spend)

  def _data(self) -> pd.DataFrame:
    data = pd.read_csv(self.data_path)
    nan_geos = data[data.isna().any(axis=1)][self.geo_column].unique()
    if nan_geos.size > 0:
      warnings.warn(
          f"The following geos have nans in the data: {nan_geos}. These geos"
          " will be removed from the input data.",
          UserWarning,
      )
    no_nan_data = data[~data[self.geo_column].isin(nan_geos)].reset_index(
        drop=True
    )
    final_week = no_nan_data[self.time_column].max()
    if self.n_dma_to_keep:
      top_dmas = (
          no_nan_data[no_nan_data[self.time_column] == final_week]
          .sort_values(by=mc.POPULATION, ascending=False)[self.geo_column]
          .values[: self.n_dma_to_keep]
      )
      top_dma_data = no_nan_data[
          no_nan_data[self.geo_column].isin(top_dmas)
      ].reset_index(drop=True)
    else:
      top_dma_data = no_nan_data

    # Collect names of non-treatment columns in the dataframe.
    fill_na_columns = [
        self.kpi_column,
        mc.POPULATION,
    ]
    if self.revenue_per_kpi_column:
      fill_na_columns.append(self.revenue_per_kpi_column)
    # There may be more than one column for each prefix.
    non_treatment_prefixes = [
        c.CONTROL,
        c.NON_MEDIA_PREFIX,
        c.SPEND,
        mc.RF_SPEND,
    ]
    if self.include_gqv:
      non_treatment_prefixes.append(c.GQV)
    for column in top_dma_data.columns:
      if any(prefix in column for prefix in non_treatment_prefixes):
        fill_na_columns.append(column)

    # Fill non-treatment columns with nans for lagged media periods.
    start_week = sorted(top_dma_data[self.time_column].unique())[
        self.n_lag_history_periods
    ]
    fill_na_idxs = np.where(top_dma_data[self.time_column] < start_week)[0]
    top_dma_data.loc[fill_na_idxs, fill_na_columns] = None
    return top_dma_data

  def _get_coord_to_columns(self) -> load.CoordToColumns:
    """Parses the data columns and returns a CoordToColumns object.

    For more information on CoordToColumns, see:
    https://source.corp.google.com/piper///depot/google3/third_party/py/meridian/data/load.py

    Returns:
      CoordToColumns mapping for DataFrameDataLoader
    """
    data_columns = self.data.columns
    controls = []
    media_channels = []
    media_spend_columns = []
    reach_channels = []
    rf_spend_columns = []
    freq_channels = []
    organic_media_columns = []
    organic_reach_columns = []
    organic_frequency_columns = []
    non_media_treatment_columns = []
    control_prefixes = [c.CONTROL, c.GQV] if self.include_gqv else [c.CONTROL]
    self.data[self.geo_column] = self.data[self.geo_column].astype(str)

    for column in data_columns:
      if mc.ORGANIC_MEDIA in column:
        organic_media_columns.append(column)
      elif mc.ORGANIC_REACH in column:
        organic_reach_columns.append(column)
      elif mc.ORGANIC_FREQUENCY in column:
        organic_frequency_columns.append(column)
      elif mc.REACH in column:
        reach_channels.append(column)
        rf_spend_columns.append(
            c.SPEND + "_" + get_channel_name_from_column(column)
        )
      elif c.IMPRESSIONS in column:
        media_channels.append(column)
        media_spend_columns.append(
            c.SPEND + "_" + get_channel_name_from_column(column)
        )
      elif c.FREQ in column:
        freq_channels.append(column)
      elif any(prefix in column for prefix in control_prefixes):
        controls.append(column)
      elif c.NON_MEDIA_PREFIX in column:
        non_media_treatment_columns.append(column)

    media_spend_columns_without_nonpaid = []
    for media_spend_column in media_spend_columns:
      channel_name = "_".join(media_spend_column.split("_")[1:])
      if not any(channel_name in column for column in organic_media_columns):
        media_spend_columns_without_nonpaid.append(media_spend_column)

    rf_spend_columns_without_nonpaid = []
    for rf_spend_column in rf_spend_columns:
      channel_name = "_".join(rf_spend_column.split("_")[1:])
      if not any(channel_name in column for column in organic_reach_columns):
        rf_spend_columns_without_nonpaid.append(rf_spend_column)

    for channel in reach_channels:
      temp_channel = channel.replace(mc.REACH, c.IMPRESSIONS)
      temp_spend_channel = channel.replace(mc.REACH, c.SPEND)
      if temp_channel in media_channels:
        media_channels.remove(temp_channel)
        media_spend_columns.remove(temp_spend_channel)
    return load.CoordToColumns(
        geo=self.geo_column,
        time=self.time_column,
        kpi=self.kpi_column,
        revenue_per_kpi=self.revenue_per_kpi_column,
        controls=controls or None,
        non_media_treatments=non_media_treatment_columns or None,
        media=media_channels or None,
        media_spend=media_spend_columns_without_nonpaid or None,
        reach=reach_channels or None,
        frequency=freq_channels or None,
        rf_spend=rf_spend_columns_without_nonpaid or None,
        organic_media=organic_media_columns or None,
        organic_reach=organic_reach_columns or None,
        organic_frequency=organic_frequency_columns or None,
    )


@dataclasses.dataclass
class ModelSpecParams:
  """Validates and holds `model.ModelSpec` parameters.

  Attributes:
    media_effects_dist: A string to specify the distribution of media random
      effects across geos. This attribute is not used with a National level
      model. Allowed values: "normal" or "log_normal". Default: "log_normal".
    hill_before_adstock: A boolean indicating whether to apply the hill function
      before the adstock function, in contrast to the default order of adstock
      before hill. This argument does not apply to RF channels. Default:
      `False`.
    max_lag: An integer indicating the maximum number of lag periods (>= 0) to
      include in the adstock calculation. May also be set to `None`, which is
      equivalent to infinite max lag. Default: 8.
    unique_sigma_for_each_geo: A boolean indicating whether to use a unique
      residual variance for each geo. If `False`, then a single residual
      variance is used for all geos. Default: `False`.
    paid_media_prior_type: A string to specify the prior type for the media
      coefficients. Allowed values: `'roi'`, `'mroi'`, `'contribution`',
      `'coefficient'`. The `PriorDistribution` contains `roi_m`, `mroi_m`,
      `contribution_m`, and`beta_m`, but only one of these is used depending on
      the `paid_media_prior_type`. Likewise, the `PriorDistribution` contains
      distributions `roi_rf`, `mroi_rf`, `contribution_rf`, and`beta_rf`, but
      only one of these is used depending on the `paid_media_prior_type`. When
      `paid_media_prior_type` is `'roi'`, the `PriorDistribution.roi_m` and
      `PriorDistribution.roi_rf` parameters are used to specify a prior on the
      ROI. When `paid_media_prior_type` is `'mroi'`, the
      `PriorDistribution.mroi_m` and `PriorDistribution.mroi_rf` parameters are
      used to specify a prior on the mROI. When `paid_media_prior_type` is
      `'contribution'`, the `PriorDistribution.contribution_m` and
      `PriorDistribution.contribution_rf` parameters are used to specify a prior
      on the contribution. When `paid_media_prior_type` is `'coefficient'`, the
      `PriorDistribution.beta_m` and `PriorDistribution.beta_rf` parameters are
      used to specify a prior on the coefficient mean parameters. Default:
      `'roi'`.
    organic_media_prior_type: A string to specify the prior type for the organic
      media coefficients. Allowed values: `'contribution'`, `'coefficient'`. The
      `PriorDistribution` contains `contribution_om` and`beta_om`, but only one
      of these is used depending on the `organic_media_prior_type`. Likewise,
      the `PriorDistribution` contains distributions `contribution_orf`,
      and`beta_orf`, but only one of these is used depending on the
      `organic_media_prior_type`. When `organic_media_prior_type` is
      `'contribution'`, the `PriorDistribution.contribution_om` and
      `PriorDistribution.contribution_orf` parameters are used to specify a
      prior on the contribution. When `organic_media_prior_type` is
      `'coefficient'`, the `PriorDistribution.beta_om` and
      `PriorDistribution.beta_orf` parameters are used to specify a prior on the
      coefficient mean parameters. Default: `'contribution'`.
    non_media_treatments_prior_type: A string to specify the prior type for the
      non-media treatment coefficients. Allowed values: `'contribution'`,
      `'coefficient'`. The `PriorDistribution` contains `contribution_n`
      and`gamma_n`, but only one of these is used depending on the
      `non_media_prior_type`.When `non_media_prior_type` is `'contribution'`,
      the `PriorDistribution.contribution_n` parameter is used to specify a
      prior on the contribution. When `non_media_prior_type` is `'coefficient'`,
      the `PriorDistribution.gamma_n` parameter is used to specify a prior on
      the coefficient mean parameters. Default: `'contribution'`.
    non_media_treatments_baseline_values: Optional list of shape
      (n_non_media_channels,). Each element is either a float (which means that
      the fixed value will be used as baseline for the given channel) or one of
      the strings "min" or "max" (which mean that the global minimum or maximum
      value will be used as baseline for the scaled values of the given
      non_media treatments channel). If None, the minimum value is used as
      baseline for each non_media treatments channel.
    knots: An optional integer or list of integers indicating the knots used to
      estimate time effects. When `knots` is a list of integers, the knot
      locations are provided by that list (zero corresponds to a knot at the
      first time period, one corresponds to a knot at the second time, ..., and
      (`n_times` - 1) corresponds to a knot at the last time period). Typically
      we suggest knots at 0 and (`n_times` - 1) are included, but this is not
      strictly necessary. When `knots` is an integer, then there are `knots`
      many knots with locations equally spaced across the time periods
      (including knots at zero and (`n_times` - 1). When `knots` is 1, there is
      a single common regression coefficient used for all time periods. If
      `knots` is set to `None`, then the numbers of knots used is equal to the
      number of time periods in the case of a geo model. This is equivalent to
      each time period having its own regression coefficient. If `knots` is set
      to `None` in the case of a national model, then the number of knots used
      is 1. Default: `None`.
    test_pct: Percentage of data to be held out of modeling for testing. Values
      must be greater than or equal to zero, but strictly less than one.
    baseline_geo: Identifies the baseline geo. The baseline geo is treated as
      the reference geo in the dummy encoding of geos. Non-baseline geos have a
      corresponding `tau_g` indicator variable which means that they have a
      higher prior variance than the baseline geo. When set to `None`, the geo
      with the biggest population is used as the baseline. Default: `None`.
    control_population_scaling_id: An optional boolean tensor of dimension
      (`n_controls`) indicating the control variables for which the control
      value should be scaled by population.
    priors: A `PriorDistribution` object specifying the prior distribution of
      each set of model parameters. Either read from `prior_path` or created
      using `custom_priors.`
  """

  media_effects_dist: str
  hill_before_adstock: bool
  max_lag: int
  unique_sigma_for_each_geo: bool
  paid_media_prior_type: str
  organic_media_prior_type: str
  non_media_treatments_prior_type: str
  non_media_treatments_baseline_values: Sequence[float | str] | None
  knots: int | Sequence[int]
  test_pct: float
  baseline_geo: int | str | None
  control_population_scaling_id: Sequence[bool] | None
  priors: prior_distribution.PriorDistribution

  def __post_init__(self):
    self._validate_args()

  def _validate_args(self):
    """Validates arguments for `ModelSpecParams`."""
    _validate_type(c.TEST_PCT, self.test_pct, float)
    _validate_type(c.PAID_MEDIA_PRIOR_TYPE, self.paid_media_prior_type, str)
    _validate_type(
        c.ORGANIC_MEDIA_PRIOR_TYPE, self.organic_media_prior_type, str
    )
    _validate_type(
        c.NON_MEDIA_TREATMENTS_PRIOR_TYPE,
        self.non_media_treatments_prior_type,
        str,
    )
    _validate_type(
        c.NON_MEDIA_TREATMENTS_BASELINE_VALUES,
        self.non_media_treatments_baseline_values,
        (Sequence, type(None)),
    )

    if not 0.0 <= self.test_pct < 1.0:
      raise ValueError(
          "test_pct must be greater than or equal to zero and less than one.",
      )


@dataclasses.dataclass
class PriorDistributionParams:
  """Validates and holds `prior_distribution.PriorDistribution` parameters.

  Example of the custom_priors argument:

  ```python
  {
    "roi_m": {
      "loc": 0.4,
      "scale": 0.4,
      "distribution_type":"LogNormal",
    },
    "ec_m": {
      "distribution": {
        "scale": 0.5,
        "distribution_type": "HalfNormal",
      },
      "bijector": {
        "shift": 0.1,
        "bijector_type": "Shift",
      },
      "distribution_type": "TransformedDistribution",
    },
  }
  ```

  As the example shows, complex distribution types such as
  `tfp.distributions.TransformedDistribution`, can be configured using
  nested maps.

  Attributes:
    prior_path: A path to a saved PriorDistribution object to be used for
      modeling.
    custom_priors: Mapping of distribution parameters. Each top level key should
      be the string name of a PriorDistribution attribute. Each key should have
      at a minimum a distribution_type string, which will be converted to the
      equivalent tfp.distributions type. The remaining key value pairs will
      depend on the type of distribution used.
    kpi: Sum of the entire KPI across geos and time.
    media_spend: Spend per media channel summed across geos and time.
  """

  prior_path: str | None
  custom_priors: _PRIOR_MAPPING_TYPES
  kpi: float
  media_spend: Sequence[float]

  def __post_init__(self):
    self._validate_args()

  @functools.cached_property
  def priors(self) -> prior_distribution.PriorDistribution:
    """Returns a PriorDistribution object based configuration attribute values.

    Returns:
      PriorDistribution for `spec.ModelSpec` object used for `meridian.model`
        model fitting.
    """
    if self.prior_path:
      return self._read_prior_path()

    custom_priors = dict(self.custom_priors) if self.custom_priors else {}

    return _create_prior_distribution(custom_priors)

  def _validate_args(self):
    """Validates arguments for `ModelSpecParams`."""
    _validate_type(c.PRIOR_PATH, self.prior_path, (str, type(None)))
    _validate_type(c.CUSTOM_PRIORS, self.custom_priors, (Mapping, type(None)))

    if self.prior_path:
      _validate_path(c.PRIOR_PATH, self.prior_path)
      if self.custom_priors:
        raise AttributeError(
            f"Only one of `{c.PRIOR_PATH}` and `{c.CUSTOM_PRIORS}` may be"
            " provided."
        )

  def _read_prior_path(self) -> prior_distribution.PriorDistribution:
    """Reads the PriorDistribution from the specified data path.

    Returns:
      Custom Prior Distribution
    """
    with gfile.Open(self.prior_path, "rb") as f:
      return joblib.load(f)


@dataclasses.dataclass(frozen=True)
class ModelingParams:
  """Validates and holds model sampling parameters.

  Attributes:
    n_adapt: Integer number of adaptation draws per chain.
    n_burnin: Integer number of burn-in draws per chain. Burn-in draws occur
      after adaptation draws and before the kept draws.
    n_chains: Integer number of MCMC chains.
    n_draws: Integer number of prior draws. If None, no prior samples will be
      drawn from the PriorDistribution.
    n_keep: Integer number of draws per chain to keep for inference.
    sample_prior: Boolean flag for running sample_prior() function.
  """

  n_adapt: int
  n_burnin: int
  n_chains: int
  n_draws: int | None
  n_keep: int

  @property
  def sample_prior(self) -> bool:
    return self.n_draws is not None

  def __post_init__(self):
    self._validate_args()

  def _validate_args(self):
    _validate_type(c.N_ADAPT, self.n_adapt, int)
    _validate_type(c.N_BURNIN, self.n_burnin, int)
    _validate_type(c.N_CHAINS, self.n_chains, (int, Sequence))
    _validate_type(c.N_DRAWS, self.n_draws, (int, type(None)))
    _validate_type(c.N_KEEP, self.n_keep, int)


@dataclasses.dataclass(frozen=True)
class BudgetConstraints:
  """Validates and holds BudgetOptimizer.optimize parameters.

  Attributes:
    use_posterior: Boolean. If `True`, then the budget is optimized based on the
      posterior distribution of the model. Otherwise, the prior distribution is
      used.
    fixed_budget: Boolean indicating whether it's a fixed budget optimization or
      flexible budget optimization. Defaults to `True`. If `False`, must specify
      either `target_roi` or `target_mroi`.
    target_roi: Float indicating the target ROI constraint. Only used for
      flexible budget scenarios. The budget is constrained to when the ROI of
      the total spend hits `target_roi`.
    target_mroi: Float indicating the target marginal ROI constraint. Only used
      for flexible budget scenarios. The budget is constrained to when the
      marginal ROI of the total spend hits `target_mroi`.
    use_optimal_frequency: If `True`, uses `optimal_frequency` calculated by
      trained Meridian model for optimization. If `False`, uses historical
      frequency.
    use_kpi: If `True`, runs the optimization on KPI. Should be consistend with
      the KPI type used in the LoaderParams.
    n_optimization_periods: The last n time periods to use for the budget
      optimization scenario.
    spend_constraint_lower: Numeric list of size `n_total_channels` or float
      (same constraint for all channels) indicating the lower bound of
      media-level spend. The lower bound of media-level spend is `(1 -
      spend_constraint_lower) * budget * allocation`. The value must be between
      0-1. Defaults to `0.3` for fixed budget and `1` for flexible.
    spend_constraint_upper: Numeric list of size `n_total_channels` or float
      (same constraint for all channels) indicating the upper bound of
      media-level spend. The upper bound of media-level spend is `(1 +
      spend_constraint_upper) * budget * allocation`. Defaults to `0.3` for
      fixed budget and `1` for flexible.
    time: Time periods in `meridian.input_data`.
    selected_times: Tuple containing the start and end time dimensions for the
      duration to run the optimization on. Start time is inferred from times and
      n_optimization_periods, and end time is always the last entry in times.
      Note: It is assumed the time labels are in sorted order.
  """

  use_posterior: bool
  fixed_budget: bool
  target_roi: float | None
  target_mroi: float | None
  use_optimal_frequency: bool
  use_kpi: bool
  n_optimization_periods: int | None
  spend_constraint_lower: Sequence[float] | float | None
  spend_constraint_upper: Sequence[float] | float | None
  time: Sequence[str]

  def __post_init__(self):
    self._validate_args()

  @property
  def selected_times(self) -> tuple[str, str] | None:
    return (
        (self.time[-self.n_optimization_periods], self.time[-1])
        if self.n_optimization_periods
        else None
    )

  def _validate_args(self):
    """Validates arguments for `BudgetConstraints`."""
    _validate_type(c.USE_POSTERIOR, self.use_posterior, bool)
    _validate_type(c.FIXED_BUDGET, self.fixed_budget, bool)
    _validate_type(c.TARGET_ROI, self.target_roi, (float, type(None)))
    _validate_type(c.TARGET_MROI, self.target_mroi, (float, type(None)))
    _validate_type(
        c.USE_OPTIMAL_FREQUENCY,
        self.use_optimal_frequency,
        bool,
    )
    _validate_type(c.USE_KPI, self.use_kpi, bool)
    _validate_type(
        c.SPEND_CONSTRAINT_LOWER,
        self.spend_constraint_lower,
        (Sequence, float, type(None)),
    )
    _validate_type(
        c.SPEND_CONSTRAINT_UPPER,
        self.spend_constraint_upper,
        (Sequence, float, type(None)),
    )
    _validate_type(
        c.N_OPTIMIZATION_PERIODS,
        self.n_optimization_periods,
        (int, type(None)),
    )
    if (
        self.n_optimization_periods is not None
        and not 1 <= self.n_optimization_periods <= len(self.time)
    ):
      raise ValueError(
          "n_optimization_periods must be in the inclusive interval [1, n_time"
          " periods]."
      )


@dataclasses.dataclass(frozen=True)
class OptimizationRuntimeStats:
  """Contains runtime stats for `opt.BudgetOptimizer.optimize()`.

  Attributes:
    optimization_results: An `OptimizationResults` object containing the output
      of `optimize()`.
    wall_elapsed_ms: Time in milliseconds it took to run `optimize()`.
    proc_elapsed_ms: Process time in milliseconds it took to run `optimize()`.
    peak_gpu_mem_gb: The peak GPU memory consumption during optimization. 0 if
      using a cpu.
  """

  optimization_results: opt.OptimizationResults
  wall_elapsed_ms: float
  proc_elapsed_ms: float
  peak_gpu_mem_gb: float


@dataclasses.dataclass(frozen=True)
class SamplingRuntimeStats:
  """Contains runtime stats for `model.Meridian` sampling functions.

  Attributes:
    meridian: Model whose sampling functions' runtime stats are recorded.
    wall_elapsed_ms: Time in milliseconds it took to run sampling functions.
    proc_elapsed_ms: Process time in milliseconds it took to run sampling
      functions.
    peak_gpu_mem_gb: The peak GPU memory consumption during sampling. 0 if using
      a cpu.
  """

  meridian: model.Meridian
  wall_elapsed_ms: float
  proc_elapsed_ms: float
  peak_gpu_mem_gb: float


def _create_input_data(config: dict[str, Any]) -> input_data.InputData:
  """Creates `meridian.data.InputData` object from config parameters.

  Args:
    config: Configuration parameters loaded from configuration file specifying
      data, model, and training parameters.

  Returns:
    Input data for `meridian.model` creation.
  """
  path_type = config.get(c.PATH_TYPE, None)
  if path_type not in c.PATH_TYPES:
    raise ValueError(f"Path type {path_type} is not supported.")

  if path_type == c.XARRAY:
    data_params = XarrayDataLoaderParams(
        data_path=config.get(c.DATA_PATH, None),
        kpi_type=config.get(c.KPI_TYPE, c.PARAM_DEFAULTS[c.KPI_TYPE]),
        path_type=path_type,
    )
    loader = load.XrDatasetDataLoader(
        dataset=data_params.data,
        kpi_type=data_params.kpi_type,
    )
  elif path_type == c.CSV:
    data_params = DataFrameDataLoaderParams(
        data_path=config.get(c.DATA_PATH, None),
        path_type=path_type,
        kpi_type=config.get(c.KPI_TYPE, c.PARAM_DEFAULTS[c.KPI_TYPE]),
        geo_column=config.get(c.GEO_COLUMN, c.PARAM_DEFAULTS[c.GEO_COLUMN]),
        time_column=config.get(c.TIME_COLUMN, c.PARAM_DEFAULTS[c.TIME_COLUMN]),
        kpi_column=config.get(c.KPI_COLUMN, c.PARAM_DEFAULTS[c.KPI_COLUMN]),
        n_dma_to_keep=config.get(
            c.N_DMA_TO_KEEP, c.PARAM_DEFAULTS[c.N_DMA_TO_KEEP]
        ),
        n_lag_history_periods=config.get(
            c.N_LAG_HISTORY_PERIODS, c.PARAM_DEFAULTS[c.N_LAG_HISTORY_PERIODS]
        ),
        include_gqv=config.get(c.INCLUDE_GQV, c.PARAM_DEFAULTS[c.INCLUDE_GQV]),
        revenue_per_kpi_column=config.get(
            c.REVENUE_PER_KPI_COLUMN, c.PARAM_DEFAULTS[c.REVENUE_PER_KPI_COLUMN]
        ),
    )
    loader = load.DataFrameDataLoader(
        df=data_params.data,
        coord_to_columns=data_params.coord_to_columns,
        kpi_type=data_params.kpi_type,
        media_to_channel=data_params.media_to_channel,
        media_spend_to_channel=data_params.media_spend_to_channel,
        reach_to_channel=data_params.reach_to_channel,
        frequency_to_channel=data_params.frequency_to_channel,
        rf_spend_to_channel=data_params.rf_spend_to_channel,
    )
  else:
    raise ValueError(f"Path type {path_type} is not supported.")
  return loader.load()


def _create_modeling_params(
    config: Mapping[str, int | None],
) -> ModelingParams:
  """Creates `ModelingParams` object from config parameters.

  Args:
    config: Configuration parameters loaded from configuration file specifying
      data, model, and training parameters.

  Returns:
    Modeling and optimization parameters for `meridian.model` creation.
  """
  return ModelingParams(
      n_adapt=config.get(c.N_ADAPT, c.PARAM_DEFAULTS[c.N_ADAPT]),
      n_burnin=config.get(c.N_BURNIN, c.PARAM_DEFAULTS[c.N_BURNIN]),
      n_chains=config.get(c.N_CHAINS, c.PARAM_DEFAULTS[c.N_CHAINS]),
      n_draws=config.get(c.N_DRAWS, c.PARAM_DEFAULTS[c.N_DRAWS]),
      n_keep=config.get(c.N_KEEP, c.PARAM_DEFAULTS[c.N_KEEP]),
  )


def _create_model_spec(
    config: dict[str, Any],
    n_geos: int,
    n_times: int,
    kpi: float,
    media_spend: Sequence[float],
) -> spec.ModelSpec:
  """Creates `spec.ModelSpec` object from config parameters.

  Args:
    config: Configuration parameters loaded from configuration file specifying
      data, model, and training parameters.
    n_geos: Number of geographical locations in the dataset.
    n_times: Number of time periods in the dataset.
    kpi: Sum of the entire KPI across geos and time.
    media_spend: Spend per media channel summed across geos and time.

  Returns:
    Model parameters for `meridian.model` creation.
  """
  prior_params = PriorDistributionParams(
      prior_path=config.get(c.PRIOR_PATH, c.MODEL_SPEC_DEFAULTS[c.PRIOR_PATH]),
      custom_priors=config.get(
          c.CUSTOM_PRIORS, c.MODEL_SPEC_DEFAULTS[c.CUSTOM_PRIORS]
      ),
      kpi=kpi,
      media_spend=media_spend,
  )
  control_population_scaling_id = config.get(
      c.CONTROL_POPULATION_SCALING_ID,
      c.MODEL_SPEC_DEFAULTS[c.CONTROL_POPULATION_SCALING_ID],
  )
  control_population_scaling_id_or_none = (
      np.array(control_population_scaling_id)
      if control_population_scaling_id is not None
      else None
  )
  spec_params = ModelSpecParams(
      media_effects_dist=config.get(
          c.MEDIA_EFFECTS_DIST, c.MODEL_SPEC_DEFAULTS[c.MEDIA_EFFECTS_DIST]
      ),
      hill_before_adstock=config.get(
          c.HILL_BEFORE_ADSTOCK, c.MODEL_SPEC_DEFAULTS[c.HILL_BEFORE_ADSTOCK]
      ),
      max_lag=config.get(c.MAX_LAG, c.MODEL_SPEC_DEFAULTS[c.MAX_LAG]),
      unique_sigma_for_each_geo=config.get(
          c.UNIQUE_SIGMA_FOR_EACH_GEO,
          c.MODEL_SPEC_DEFAULTS[c.UNIQUE_SIGMA_FOR_EACH_GEO],
      ),
      paid_media_prior_type=config.get(
          c.PAID_MEDIA_PRIOR_TYPE,
          c.MODEL_SPEC_DEFAULTS[c.PAID_MEDIA_PRIOR_TYPE],
      ),
      organic_media_prior_type=config.get(
          c.ORGANIC_MEDIA_PRIOR_TYPE,
          c.MODEL_SPEC_DEFAULTS[c.ORGANIC_MEDIA_PRIOR_TYPE],
      ),
      non_media_treatments_prior_type=config.get(
          c.NON_MEDIA_TREATMENTS_PRIOR_TYPE,
          c.MODEL_SPEC_DEFAULTS[c.NON_MEDIA_TREATMENTS_PRIOR_TYPE],
      ),
      non_media_treatments_baseline_values=config.get(
          c.NON_MEDIA_TREATMENTS_BASELINE_VALUES,
          c.MODEL_SPEC_DEFAULTS[c.NON_MEDIA_TREATMENTS_BASELINE_VALUES],
      ),
      knots=config.get(c.KNOTS, c.MODEL_SPEC_DEFAULTS[c.KNOTS]),
      test_pct=config.get(c.TEST_PCT, c.MODEL_SPEC_DEFAULTS[c.TEST_PCT]),
      baseline_geo=config.get(
          c.BASELINE_GEO, c.MODEL_SPEC_DEFAULTS[c.BASELINE_GEO]
      ),
      control_population_scaling_id=control_population_scaling_id_or_none,
      priors=prior_params.priors,
  )

  np.random.seed(0)
  holdout_id = np.full([n_geos, n_times], False)
  test_samples_per_time = np.round(spec_params.test_pct * n_times).astype(int)
  holdout_id[:, :test_samples_per_time] = True
  holdout_id = np.random.default_rng().permuted(holdout_id, axis=1)
  if n_geos == 1:
    holdout_id = np.squeeze(holdout_id, axis=0)

  return spec.ModelSpec(
      prior=spec_params.priors,
      media_effects_dist=spec_params.media_effects_dist,
      hill_before_adstock=spec_params.hill_before_adstock,
      max_lag=spec_params.max_lag,
      unique_sigma_for_each_geo=spec_params.unique_sigma_for_each_geo,
      media_prior_type=spec_params.paid_media_prior_type,
      organic_media_prior_type=spec_params.organic_media_prior_type,
      non_media_treatments_prior_type=spec_params.non_media_treatments_prior_type,
      non_media_treatments_baseline_values=spec_params.non_media_treatments_baseline_values,
      knots=spec_params.knots,
      baseline_geo=spec_params.baseline_geo,
      holdout_id=holdout_id,
      control_population_scaling_id=spec_params.control_population_scaling_id,
  )


def _create_optimization_params(
    config: Mapping[str, bool | int | None],
    time_labels: Sequence[str],
) -> BudgetConstraints:
  """Creates `BudgetConstraints` object from config parameters.

  Args:
    config: Configuration parameters loaded from configuration file specifying
      data, model, and training parameters.
    time_labels: Time periods from `meridian.input_data`.

  Returns:
    Optimization parameters for `optimizer.BudgetOptimizer.optimize()`.
  """
  return BudgetConstraints(
      use_posterior=config.get(
          c.USE_POSTERIOR, c.PARAM_DEFAULTS[c.USE_POSTERIOR]
      ),
      fixed_budget=bool(
          config.get(c.FIXED_BUDGET, c.PARAM_DEFAULTS[c.FIXED_BUDGET])
      ),
      target_roi=config.get(c.TARGET_ROI, c.PARAM_DEFAULTS[c.TARGET_ROI]),
      target_mroi=config.get(c.TARGET_MROI, c.PARAM_DEFAULTS[c.TARGET_MROI]),
      use_optimal_frequency=config.get(
          c.USE_OPTIMAL_FREQUENCY, c.PARAM_DEFAULTS[c.USE_OPTIMAL_FREQUENCY]
      ),
      use_kpi=config.get(c.USE_KPI, c.PARAM_DEFAULTS[c.USE_KPI]),
      n_optimization_periods=config.get(
          c.N_OPTIMIZATION_PERIODS, c.PARAM_DEFAULTS[c.N_OPTIMIZATION_PERIODS]
      ),
      spend_constraint_lower=config.get(
          c.SPEND_CONSTRAINT_LOWER, c.PARAM_DEFAULTS[c.SPEND_CONSTRAINT_LOWER]
      ),
      spend_constraint_upper=config.get(
          c.SPEND_CONSTRAINT_UPPER, c.PARAM_DEFAULTS[c.SPEND_CONSTRAINT_UPPER]
      ),
      time=time_labels,
  )


def _get_peak_gpu_memory_gb() -> float:
  if tf.config.list_physical_devices("GPU"):
    peak_gpu_mem_gb = tf.config.experimental.get_memory_info("GPU:0")[
        "peak"
    ] / (1000**3)
    tf.config.experimental.reset_memory_stats("GPU:0")
  else:
    peak_gpu_mem_gb = 0.0
  return peak_gpu_mem_gb


def optimize_and_record(
    optimizer: opt.BudgetOptimizer,
    optimization_params: BudgetConstraints,
) -> OptimizationRuntimeStats:
  """Optimize `meridian` spend allocation using `optimizer.BudgetOptimizer`.

  Args:
    optimizer: `meridian` inialized `opt.BudgetOptimizer` for budget allocation
      optimization.
    optimization_params: Parameters used to define optimization scenario.

  Returns:
    runtime_stats: Optimize runtime stats with `opt.BudgetOptimizer` whose
      `optimize()` method was called.
    .
  """
  st = time.time()
  pst = time.process_time()
  optimization_results = optimizer.optimize(
      use_posterior=optimization_params.use_posterior,
      fixed_budget=optimization_params.fixed_budget,
      target_roi=optimization_params.target_roi,
      target_mroi=optimization_params.target_mroi,
      use_optimal_frequency=optimization_params.use_optimal_frequency,
      use_kpi=optimization_params.use_kpi,
      selected_times=optimization_params.selected_times,
      spend_constraint_lower=optimization_params.spend_constraint_lower,
      spend_constraint_upper=optimization_params.spend_constraint_upper,
  )
  wall_elapsed_ms = (time.time() - st) * 1000
  proc_elapsed_ms = (time.process_time() - pst) * 1000
  return OptimizationRuntimeStats(
      optimization_results,
      wall_elapsed_ms,
      proc_elapsed_ms,
      _get_peak_gpu_memory_gb(),
  )


def fit_and_record(
    mmm: model.Meridian,
    training_params: ModelingParams,
) -> SamplingRuntimeStats:
  """Call `model.Meridian` sampling functions.

  Args:
    mmm: Model used for parameter fitting.
    training_params: Parameters used to define MCMC sampling arguments.

  Returns:
    runtime_stats: Sampling runtime stats with `model.Meridian` whose sampling
      methods were called.
  """
  st = time.time()
  pst = time.process_time()
  if training_params.sample_prior:
    mmm.sample_prior(training_params.n_draws)
  mmm.sample_posterior(
      n_chains=training_params.n_chains,
      n_adapt=training_params.n_adapt,
      n_burnin=training_params.n_burnin,
      n_keep=training_params.n_keep,
  )
  wall_elapsed_ms = (time.time() - st) * 1000
  proc_elapsed_ms = (time.process_time() - pst) * 1000
  return SamplingRuntimeStats(
      mmm, wall_elapsed_ms, proc_elapsed_ms, _get_peak_gpu_memory_gb()
  )


def save_mmm(mmm: model.Meridian, session_path: str):
  """Save the model object to a `pickle` file path.

  Args:
    mmm: Model object to save.
    session_path: File path to save a pickled model object.
  """
  file_path = os.path.join(session_path, c.MODEL_FILE_NAME)
  if not gfile.Exists(file_path):
    gfile.MakeDirs(
        os.path.dirname(file_path),
        mode=gfile.LEGACY_GROUP_WRITABLE_WORLD_READABLE,
    )

  with gfile.Open(file_path, "wb") as f:
    joblib.dump(mmm, f)


def save_optimizer_data(
    optimization_results: opt.OptimizationResults, session_path: str
):
  """Save optimizer data to `session_path`.

  Args:
    optimization_results: OptimizationResults data to save.
    session_path: Directory to save optimizer data to.
  """
  file_path = os.path.join(session_path, c.OPTIMIZER_DATA_FILE_NAME)
  optimizer_data = utils.OptimizerData(
      nonoptimized_data=optimization_results.nonoptimized_data,
      nonoptimized_data_with_optimal_freq=optimization_results.nonoptimized_data_with_optimal_freq,
      optimized_data=optimization_results.optimized_data,
  )
  utils.save_optimizer_data(optimizer_data, file_path)


def main(_) -> None:
  # pylint: disable=unused-variable
  ldap = _LDAP.value
  config_path_str = _CONFIG.value
  user_path = utils.get_user_path(ldap)
  config_path = gpath.GPath(config_path_str)
  config = utils.get_config(config_path)
  session_name = config.get(c.SESSION_NAME, c.PARAM_DEFAULTS[c.SESSION_NAME])
  session_path = utils.get_session_path(user_path, session_name)

  data = _create_input_data(config)
  model_spec = _create_model_spec(
      config,
      len(data.kpi.coords[mc.GEO]),
      len(data.kpi.coords[mc.TIME]),
      kpi=np.sum(data.kpi.values),
      media_spend=np.sum(data.get_total_spend(), axis=(0, 1)),
  )
  training_params = _create_modeling_params(config)
  mmm = model.Meridian(input_data=data, model_spec=model_spec)

  sampling_runtime_stats = fit_and_record(mmm, training_params)
  save_mmm(sampling_runtime_stats.meridian, session_path)

  optimizer = opt.BudgetOptimizer(sampling_runtime_stats.meridian)
  opt_runtime_stats = optimize_and_record(
      optimizer,
      _create_optimization_params(config, mmm.input_data.time.values),
  )
  save_optimizer_data(opt_runtime_stats.optimization_results, session_path)

  runtime_stats = {
      c.USE_GPU.lower(): _USE_GPU,
      c.N_GPUS.lower(): _N_GPUS,
      c.N_CPUS.lower(): _N_CPUS,
      c.RAM_GIB.lower(): _RAM_GIB,
      c.SAMPLING_WALL_ELAPSED_MS: sampling_runtime_stats.wall_elapsed_ms,
      c.SAMPLING_PROCESS_ELAPSED_MS: sampling_runtime_stats.proc_elapsed_ms,
      c.SAMPLING_PEAK_GPU_MEM_GB: sampling_runtime_stats.peak_gpu_mem_gb,
      c.OPTIMIZATION_WALL_ELAPSED_TIME: opt_runtime_stats.wall_elapsed_ms,
      c.OPTIMIZATION_PROCESS_ELAPSED_TIME: opt_runtime_stats.proc_elapsed_ms,
      c.OPTIMIZATION_PEAK_GPU_MEM_GB: opt_runtime_stats.peak_gpu_mem_gb,
  }
  utils.save_runtime_stats(session_path, runtime_stats)


if __name__ == "__main__":
  app.run(main)
