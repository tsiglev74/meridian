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

"""Module for MCMC sampling of posterior distributions in a Meridian model."""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import arviz as az
from meridian import constants
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

if TYPE_CHECKING:
  from meridian.model import model  # pylint: disable=g-bad-import-order,g-import-not-at-top


__all__ = [
    "MCMCSamplingError",
    "MCMCOOMError",
    "PosteriorMCMCSampler",
]


class MCMCSamplingError(Exception):
  """The Markov Chain Monte Carlo (MCMC) sampling failed."""


class MCMCOOMError(Exception):
  """The Markov Chain Monte Carlo (MCMC) sampling exceeds memory limits."""


def _get_tau_g(
    tau_g_excl_baseline: tf.Tensor, baseline_geo_idx: int
) -> tfp.distributions.Distribution:
  """Computes `tau_g` from `tau_g_excl_baseline`.

  This function computes `tau_g` by inserting a column of zeros at the
  `baseline_geo` position in `tau_g_excl_baseline`.

  Args:
    tau_g_excl_baseline: A tensor of shape `[..., n_geos - 1]` for the
      user-defined dimensions of the `tau_g` parameter distribution.
    baseline_geo_idx: The index of the baseline geo to be set to zero.

  Returns:
    A tensor of shape `[..., n_geos]` with the final distribution of the `tau_g`
    parameter with zero at position `baseline_geo_idx` and matching
    `tau_g_excl_baseline` elsewhere.
  """
  rank = len(tau_g_excl_baseline.shape)
  shape = tau_g_excl_baseline.shape[:-1] + [1] if rank != 1 else 1
  tau_g = tf.concat(
      [
          tau_g_excl_baseline[..., :baseline_geo_idx],
          tf.zeros(shape, dtype=tau_g_excl_baseline.dtype),
          tau_g_excl_baseline[..., baseline_geo_idx:],
      ],
      axis=rank - 1,
  )
  return tfp.distributions.Deterministic(tau_g, name="tau_g")


@tf.function(autograph=False, jit_compile=True)
def _xla_windowed_adaptive_nuts(**kwargs):
  """XLA wrapper for windowed_adaptive_nuts."""
  return tfp.experimental.mcmc.windowed_adaptive_nuts(**kwargs)


class PosteriorMCMCSampler:
  """A callable that samples from posterior distributions using MCMC."""

  def __init__(self, meridian: "model.Meridian"):
    self._meridian = meridian

  def _get_joint_dist_unpinned(self) -> tfp.distributions.Distribution:
    """Returns a `JointDistributionCoroutineAutoBatched` function for MCMC."""
    mmm = self._meridian
    mmm.populate_cached_properties()

    # This lists all the derived properties and states of this Meridian object
    # that are referenced by the joint distribution coroutine.
    # That is, these are the list of captured parameters.
    prior_broadcast = mmm.prior_broadcast
    baseline_geo_idx = mmm.baseline_geo_idx
    knot_info = mmm.knot_info
    n_geos = mmm.n_geos
    n_times = mmm.n_times
    n_media_channels = mmm.n_media_channels
    n_rf_channels = mmm.n_rf_channels
    n_organic_media_channels = mmm.n_organic_media_channels
    n_organic_rf_channels = mmm.n_organic_rf_channels
    n_controls = mmm.n_controls
    n_non_media_channels = mmm.n_non_media_channels
    holdout_id = mmm.holdout_id
    media_tensors = mmm.media_tensors
    rf_tensors = mmm.rf_tensors
    organic_media_tensors = mmm.organic_media_tensors
    organic_rf_tensors = mmm.organic_rf_tensors
    controls_scaled = mmm.controls_scaled
    non_media_treatments_scaled = mmm.non_media_treatments_scaled
    media_effects_dist = mmm.media_effects_dist
    adstock_hill_media_fn = mmm.adstock_hill_media
    adstock_hill_rf_fn = mmm.adstock_hill_rf
    get_roi_prior_beta_m_value_fn = (
        mmm.prior_sampler_callable.get_roi_prior_beta_m_value
    )
    get_roi_prior_beta_rf_value_fn = (
        mmm.prior_sampler_callable.get_roi_prior_beta_rf_value
    )

    @tfp.distributions.JointDistributionCoroutineAutoBatched
    def joint_dist_unpinned():
      # Sample directly from prior.
      knot_values = yield prior_broadcast.knot_values
      gamma_c = yield prior_broadcast.gamma_c
      xi_c = yield prior_broadcast.xi_c
      sigma = yield prior_broadcast.sigma

      tau_g_excl_baseline = yield tfp.distributions.Sample(
          prior_broadcast.tau_g_excl_baseline,
          name=constants.TAU_G_EXCL_BASELINE,
      )
      tau_g = yield _get_tau_g(
          tau_g_excl_baseline=tau_g_excl_baseline,
          baseline_geo_idx=baseline_geo_idx,
      )
      mu_t = yield tfp.distributions.Deterministic(
          tf.einsum(
              "k,kt->t",
              knot_values,
              tf.convert_to_tensor(knot_info.weights),
          ),
          name=constants.MU_T,
      )

      tau_gt = tau_g[:, tf.newaxis] + mu_t
      combined_media_transformed = tf.zeros(
          shape=(n_geos, n_times, 0), dtype=tf.float32
      )
      combined_beta = tf.zeros(shape=(n_geos, 0), dtype=tf.float32)
      if media_tensors.media is not None:
        alpha_m = yield prior_broadcast.alpha_m
        ec_m = yield prior_broadcast.ec_m
        eta_m = yield prior_broadcast.eta_m
        slope_m = yield prior_broadcast.slope_m
        beta_gm_dev = yield tfp.distributions.Sample(
            tfp.distributions.Normal(0, 1),
            [n_geos, n_media_channels],
            name=constants.BETA_GM_DEV,
        )
        media_transformed = adstock_hill_media_fn(
            media=media_tensors.media_scaled,
            alpha=alpha_m,
            ec=ec_m,
            slope=slope_m,
        )
        prior_type = mmm.model_spec.paid_media_prior_type
        if isinstance(prior_type, list):
          # Per-channel prior types
          beta_m_values = []
          for i, pt in enumerate(prior_type):
            if pt in constants.PAID_MEDIA_ROI_PRIOR_TYPES:
              if pt == constants.PAID_MEDIA_PRIOR_TYPE_ROI:
                roi_or_mroi_m = yield prior_broadcast.roi_m[..., i]
              else:
                roi_or_mroi_m = yield prior_broadcast.mroi_m[..., i]
              beta_m_value = get_roi_prior_beta_m_value_fn(
                  alpha_m[..., i],
                  beta_gm_dev[:, i],
                  ec_m[..., i],
                  eta_m[..., i],
                  roi_or_mroi_m,
                  slope_m[..., i],
                  media_transformed[..., i],
              )
            else:
              beta_m_value = prior_broadcast.beta_m[..., i]
            beta_m_values.append(beta_m_value)
          beta_m = yield tfp.distributions.Deterministic(
              tf.stack(beta_m_values, axis=-1), name=constants.BETA_M
          )
        elif prior_type in constants.PAID_MEDIA_ROI_PRIOR_TYPES:
          if prior_type == constants.PAID_MEDIA_PRIOR_TYPE_ROI:
            roi_or_mroi_m = yield prior_broadcast.roi_m
          else:
            roi_or_mroi_m = yield prior_broadcast.mroi_m
          beta_m_value = get_roi_prior_beta_m_value_fn(
              alpha_m,
              beta_gm_dev,
              ec_m,
              eta_m,
              roi_or_mroi_m,
              slope_m,
              media_transformed,
          )
          beta_m = yield tfp.distributions.Deterministic(
              beta_m_value, name=constants.BETA_M
          )
        else:
          beta_m = yield prior_broadcast.beta_m

        beta_eta_combined = beta_m + eta_m * beta_gm_dev
        beta_gm_value = (
            beta_eta_combined
            if media_effects_dist == constants.MEDIA_EFFECTS_NORMAL
            else tf.math.exp(beta_eta_combined)
        )
        beta_gm = yield tfp.distributions.Deterministic(
            beta_gm_value, name=constants.BETA_GM
        )
        combined_media_transformed = tf.concat(
            [combined_media_transformed, media_transformed], axis=-1
        )
        combined_beta = tf.concat([combined_beta, beta_gm], axis=-1)

      if rf_tensors.reach is not None:
        alpha_rf = yield prior_broadcast.alpha_rf
        ec_rf = yield prior_broadcast.ec_rf
        eta_rf = yield prior_broadcast.eta_rf
        slope_rf = yield prior_broadcast.slope_rf
        beta_grf_dev = yield tfp.distributions.Sample(
            tfp.distributions.Normal(0, 1),
            [n_geos, n_rf_channels],
            name=constants.BETA_GRF_DEV,
        )
        rf_transformed = adstock_hill_rf_fn(
            reach=rf_tensors.reach_scaled,
            frequency=rf_tensors.frequency,
            alpha=alpha_rf,
            ec=ec_rf,
            slope=slope_rf,
        )

         prior_type = mmm.model_spec.rf_prior_type
         if isinstance(prior_type, list):
           # Per-channel prior types
           beta_rf_values = []
           for i, pt in enumerate(prior_type):
             if pt in constants.PAID_MEDIA_ROI_PRIOR_TYPES:
               if pt == constants.PAID_MEDIA_PRIOR_TYPE_ROI:
                 roi_or_mroi_rf = yield prior_broadcast.roi_rf[..., i]
               else:
                 roi_or_mroi_rf = yield prior_broadcast.mroi_rf[..., i]
               beta_rf_value = get_roi_prior_beta_rf_value_fn(
                   alpha_rf[..., i],
                   beta_grf_dev[:, i],
                   ec_rf[..., i],
                   eta_rf[..., i],
                   roi_or_mroi_rf,
                   slope_rf[..., i],
                   rf_transformed[..., i],
               )
             else:
               beta_rf_value = prior_broadcast.beta_rf[..., i]
             beta_rf_values.append(beta_rf_value)
           beta_rf = yield tfp.distributions.Deterministic(
               tf.stack(beta_rf_values, axis=-1),
               name=constants.BETA_RF,
           )
         elif prior_type in constants.PAID_MEDIA_ROI_PRIOR_TYPES:
          if prior_type == constants.PAID_MEDIA_PRIOR_TYPE_ROI:
            roi_or_mroi_rf = yield prior_broadcast.roi_rf
          else:
            roi_or_mroi_rf = yield prior_broadcast.mroi_rf
          beta_rf_value = get_roi_prior_beta_rf_value_fn(
              alpha_rf,
              beta_grf_dev,
              ec_rf,
              eta_rf,
              roi_or_mroi_rf,
              slope_rf,
              rf_transformed,
          )
          beta_rf = yield tfp.distributions.Deterministic(
              beta_rf_value,
              name=constants.BETA_RF,
          )
        else:
          beta_rf = yield prior_broadcast.beta_rf

        beta_eta_combined = beta_rf + eta_rf * beta_grf_dev
        beta_grf_value = (
            beta_eta_combined
            if media_effects_dist == constants.MEDIA_EFFECTS_NORMAL
            else tf.math.exp(beta_eta_combined)
        )
        beta_grf = yield tfp.distributions.Deterministic(
            beta_grf_value, name=constants.BETA_GRF
        )
        combined_media_transformed = tf.concat(
            [combined_media_transformed, rf_transformed], axis=-1
        )
        combined_beta = tf.concat([combined_beta, beta_grf], axis=-1)

      if organic_media_tensors.organic_media is not None:
        alpha_om = yield prior_broadcast.alpha_om
        ec_om = yield prior_broadcast.ec_om
        eta_om = yield prior_broadcast.eta_om
        slope_om = yield prior_broadcast.slope_om
        beta_gom_dev = yield tfp.distributions.Sample(
            tfp.distributions.Normal(0, 1),
            [n_geos, n_organic_media_channels],
            name=constants.BETA_GOM_DEV,
        )
        organic_media_transformed = adstock_hill_media_fn(
            media=organic_media_tensors.organic_media_scaled,
            alpha=alpha_om,
            ec=ec_om,
            slope=slope_om,
        )
        prior_type = mmm.model_spec.organic_media_prior_type
        if isinstance(prior_type, list):
          # Per-channel prior types
          beta_om_values = []
          for i, pt in enumerate(prior_type):
            if pt in constants.PAID_MEDIA_ROI_PRIOR_TYPES:  # Assuming PAID_MEDIA_ROI_PRIOR_TYPES is also valid for organic media
              if pt == constants.PAID_MEDIA_PRIOR_TYPE_ROI:
                roi_or_mroi_om = yield prior_broadcast.roi_om[..., i]
              else:
                roi_or_mroi_om = yield prior_broadcast.mroi_om[..., i]
              beta_om_value = get_roi_prior_beta_m_value_fn(  # Assuming get_roi_prior_beta_m_value_fn is also valid for organic media
                  alpha_om[..., i],
                  beta_gom_dev[:, i],
                  ec_om[..., i],
                  eta_om[..., i],
                  roi_or_mroi_om,
                  slope_om[..., i],
                  organic_media_transformed[..., i],
              )
            else:
              beta_om_value = prior_broadcast.beta_om[..., i]
            beta_om_values.append(beta_om_value)
          beta_om = yield tfp.distributions.Deterministic(
              tf.stack(beta_om_values, axis=-1), name=constants.BETA_OM
          )
        else:
          beta_om = yield prior_broadcast.beta_om

        beta_eta_combined = beta_om + eta_om * beta_gom_dev
        beta_gom_value = (
            beta_eta_combined
            if media_effects_dist == constants.MEDIA_EFFECTS_NORMAL
            else tf.math.exp(beta_eta_combined)
        )
        beta_gom = yield tfp.distributions.Deterministic(
            beta_gom_value, name=constants.BETA_GOM
        )
        combined_media_transformed = tf.concat(
            [combined_media_transformed, organic_media_transformed], axis=-1
        )
        combined_beta = tf.concat([combined_beta, beta_gom], axis=-1)

      if organic_rf_tensors.organic_reach is not None:
        alpha_orf = yield prior_broadcast.alpha_orf
        ec_orf = yield prior_broadcast.ec_orf
        eta_orf = yield prior_broadcast.eta_orf
        slope_orf = yield prior_broadcast.slope_orf
        beta_gorf_dev = yield tfp.distributions.Sample(
            tfp.distributions.Normal(0, 1),
            [n_geos, n_organic_rf_channels],
            name=constants.BETA_GORF_DEV,
        )
        organic_rf_transformed = adstock_hill_rf_fn(
            reach=organic_rf_tensors.organic_reach_scaled,
            frequency=organic_rf_tensors.organic_frequency,
            alpha=alpha_orf,
            ec=ec_orf,
            slope=slope_orf,
        )
+        prior_type = mmm.model_spec.organic_rf_prior_type
+        if isinstance(prior_type, list):
+          # Per-channel prior types
+          beta_orf_values = []
+          for i, pt in enumerate(prior_type):
+            if pt in constants.PAID_MEDIA_ROI_PRIOR_TYPES:  # Assuming PAID_MEDIA_ROI_PRIOR_TYPES is also valid for organic rf
+              if pt == constants.PAID_MEDIA_PRIOR_TYPE_ROI:
+                roi_or_mroi_orf = yield prior_broadcast.roi_orf[..., i]
+              else:
+                roi_or_mroi_orf = yield prior_broadcast.mroi_orf[..., i]
+              beta_orf_value = get_roi_prior_beta_rf_value_fn(  # Assuming get_roi_prior_beta_rf_value_fn is also valid for organic rf
+                  alpha_orf[..., i],
+                  beta_gorf_dev[:, i],
+                  ec_orf[..., i],
+                  eta_orf[..., i],
+                  roi_or_mroi_orf,
+                  slope_orf[..., i],
+                  organic_rf_transformed[..., i],
+              )
+            else:
+              beta_orf_value = prior_broadcast.beta_orf[..., i]
+            beta_orf_values.append(beta_orf_value)
+          beta_orf = yield tfp.distributions.Deterministic(
+              tf.stack(beta_orf_values, axis=-1), name=constants.BETA_ORF
+          )
+        else:
+          beta_orf = yield prior_broadcast.beta_orf

+        beta_eta_combined = beta_orf + eta_orf * beta_gorf_dev
+        beta_gorf_value = (
+            beta_eta_combined
+            if media_effects_dist == constants.MEDIA_EFFECTS_NORMAL
+            else tf.math.exp(beta_eta_combined)
+        )
+        beta_gorf = yield tfp.distributions.Deterministic(
            beta_gorf_value, name=constants.BETA_GORF
        )
        combined_media_transformed = tf.concat(

[end of meridian/model/posterior_sampler.py]
