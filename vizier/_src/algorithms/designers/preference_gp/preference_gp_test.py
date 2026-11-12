# Copyright 2024 Google LLC.
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

from absl.testing import absltest
from absl.testing import parameterized

import jax
jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp
import numpy as np

from vizier._src.algorithms.designers.preference_gp import preference_gp as pg


def _continuous_features(points: jax.Array) -> pg.PreferenceFeatures:
  return pg.PreferenceFeatures(
      continuous=points,
      categorical=jnp.zeros((points.shape[0], 0), dtype=jnp.int32),
  )


def _synthetic_preferences(
    num_points: int = 12,
    num_comparisons: int = 60,
    noise_sd: float = 0.1,
    rng: jax.Array = jax.random.PRNGKey(0),
):
  """Generates noisy win/loss pairs from a known 1D utility function."""
  points = jnp.linspace(0.0, 1.0, num_points).reshape(-1, 1)
  utility = jnp.sin(3.0 * points).reshape(-1)
  rng, k1, k2, k3 = jax.random.split(rng, 4)
  winner = jax.random.randint(k1, (num_comparisons,), 0, num_points)
  loser = jax.random.randint(k2, (num_comparisons,), 0, num_points)
  noisy_diff = (
      utility[winner]
      - utility[loser]
      + noise_sd * jnp.sqrt(2.0) * jax.random.normal(k3, (num_comparisons,))
  )
  keep = noisy_diff != 0
  winner, loser, noisy_diff = winner[keep], loser[keep], noisy_diff[keep]
  flipped = noisy_diff < 0
  winner_idx = jnp.where(flipped, loser, winner)
  loser_idx = jnp.where(flipped, winner, loser)
  return _continuous_features(points), utility, winner_idx, loser_idx


class PreferenceGaussianProcessTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    (
        self.features,
        self.utility,
        self.winner_idx,
        self.loser_idx,
    ) = _synthetic_preferences(rng=jax.random.PRNGKey(0))
    self.state = pg.fit_preference_gp(
        self.features,
        self.winner_idx,
        self.loser_idx,
        dim_continuous=1,
        dim_categorical=0,
        rng=jax.random.PRNGKey(1),
        num_restarts=2,
    )

  def test_win_probability_direction_and_symmetry(self):
    # High-utility point (index 9) vs low-utility point (index 2).
    features_a = _continuous_features(self.features.continuous[9:10])
    features_b = _continuous_features(self.features.continuous[2:3])
    p_ab = pg.win_probability(self.state, features_a, features_b)
    p_ba = pg.win_probability(self.state, features_b, features_a)
    self.assertGreater(float(p_ab[0]), 0.5)
    np.testing.assert_allclose(
        np.asarray(p_ab) + np.asarray(p_ba), np.ones(1), atol=1e-6
    )
    self.assertTrue(bool(jnp.all(p_ab > 0.0)))
    self.assertTrue(bool(jnp.all(p_ab < 1.0)))

  def test_utility_recovers_synthetic_trend(self):
    mean, std = pg.utility_mean_std(self.state, self.features)
    self.assertTrue(bool(jnp.all(jnp.isfinite(mean))))
    self.assertTrue(bool(jnp.all(jnp.isfinite(std))))
    correlation = jnp.corrcoef(mean, self.utility)[0, 1]
    self.assertGreater(float(correlation), 0.9)

  def test_posterior_uncertainty_shrinks_against_prior(self):
    _, std = pg.utility_mean_std(self.state, self.features)
    prior_std = float(jnp.sqrt(jnp.exp(self.state.params.log_signal_variance)))
    self.assertTrue(bool(jnp.all(std < prior_std)))
    # Posterior covariance must be positive (semi-)definite.
    _, cov = pg.posterior(self.state, self.features)
    min_eig = float(jnp.linalg.eigvalsh(cov).min())
    self.assertGreater(min_eig, 0.0)

  def test_duplicate_parameter_rows_are_safe(self):
    # Only three unique rows, but many repeated comparisons over them.
    features = _continuous_features(jnp.array([[0.1], [0.5], [0.9]]))
    winner = jnp.array([0, 0, 1, 2, 0, 2, 1, 0], dtype=jnp.int32)
    loser = jnp.array([1, 1, 2, 1, 1, 0, 0, 1], dtype=jnp.int32)
    state = pg.fit_preference_gp(
        features,
        winner,
        loser,
        dim_continuous=1,
        dim_categorical=0,
        rng=jax.random.PRNGKey(2),
        num_restarts=1,
    )
    self.assertEqual(state.num_points, 3)
    mean, std = pg.utility_mean_std(state, features)
    self.assertTrue(bool(jnp.all(jnp.isfinite(mean))))
    self.assertTrue(bool(jnp.all(jnp.isfinite(std))))

  def test_contradictory_comparisons_stay_calibrated(self):
    features = _continuous_features(jnp.array([[0.2], [0.8]]))
    # Equal evidence in both directions for the same pair.
    winner = jnp.array([0, 0, 0, 1, 1, 1], dtype=jnp.int32)
    loser = jnp.array([1, 1, 1, 0, 0, 0], dtype=jnp.int32)
    state = pg.fit_preference_gp(
        features,
        winner,
        loser,
        dim_continuous=1,
        dim_categorical=0,
        rng=jax.random.PRNGKey(3),
        num_restarts=2,
    )
    features_a = _continuous_features(features.continuous[0:1])
    features_b = _continuous_features(features.continuous[1:2])
    probability = float(pg.win_probability(state, features_a, features_b)[0])
    self.assertTrue(np.isfinite(probability))
    self.assertAlmostEqual(probability, 0.5, delta=0.1)

  def test_optimization_reduces_marginal_likelihood(self):
    initial = pg.PreferenceGPParams.initial(1, 0)
    nll_initial = pg._neg_log_marginal_likelihood(
        initial,
        self.features,
        self.winner_idx,
        self.loser_idx,
        dim_continuous=1,
        dim_categorical=0,
        num_newton_iters=20,
    )
    nll_optimized = pg._neg_log_marginal_likelihood(
        self.state.params,
        self.features,
        self.winner_idx,
        self.loser_idx,
        dim_continuous=1,
        dim_categorical=0,
        num_newton_iters=20,
    )
    self.assertLessEqual(float(nll_optimized), float(nll_initial))
    learned = self.state.params
    self.assertGreater(float(learned.noise_variance), 0.0)
    self.assertGreater(
        float(jnp.exp(learned.log_signal_variance)), 0.0
    )
    self.assertTrue(
        bool(jnp.all(jnp.exp(learned.log_continuous_length_scale) > 0.0))
    )

  def test_empty_data_returns_prior(self):
    empty = pg.PreferenceFeatures.empty(1, 0)
    state = pg.fit_preference_gp(
        empty,
        jnp.array([], dtype=jnp.int32),
        jnp.array([], dtype=jnp.int32),
        dim_continuous=1,
        dim_categorical=0,
        rng=jax.random.PRNGKey(4),
    )
    mean, cov = pg.posterior(state, self.features)
    self.assertTrue(bool(jnp.all(mean == 0.0)))
    self.assertTrue(bool(jnp.all(jnp.isfinite(cov))))
    features_a = _continuous_features(self.features.continuous[0:1])
    features_b = _continuous_features(self.features.continuous[1:2])
    probability = float(pg.win_probability(state, features_a, features_b)[0])
    self.assertAlmostEqual(probability, 0.5)

  def test_single_point_and_self_comparisons_are_finite(self):
    features = _continuous_features(jnp.array([[0.5]]))
    state = pg.fit_preference_gp(
        features,
        jnp.array([0, 0], dtype=jnp.int32),
        jnp.array([0, 0], dtype=jnp.int32),
        dim_continuous=1,
        dim_categorical=0,
        rng=jax.random.PRNGKey(5),
        num_restarts=1,
    )
    mean, std = pg.utility_mean_std(state, features)
    self.assertTrue(bool(jnp.all(jnp.isfinite(mean))))
    self.assertTrue(bool(jnp.all(jnp.isfinite(std))))

  def test_categorical_and_mixed_kernels(self):
    categorical_features = pg.PreferenceFeatures(
        continuous=jnp.zeros((4, 0)),
        categorical=jnp.array([[0], [1], [2], [0]], dtype=jnp.int32),
    )
    winner = jnp.array([0, 1, 2], dtype=jnp.int32)
    loser = jnp.array([1, 2, 3], dtype=jnp.int32)
    categorical_state = pg.fit_preference_gp(
        categorical_features,
        winner,
        loser,
        dim_continuous=0,
        dim_categorical=1,
        rng=jax.random.PRNGKey(6),
        num_restarts=1,
    )
    mean, std = pg.utility_mean_std(
        categorical_state, categorical_features
    )
    self.assertTrue(bool(jnp.all(jnp.isfinite(mean))))
    self.assertTrue(bool(jnp.all(jnp.isfinite(std))))

    mixed_features = pg.PreferenceFeatures(
        continuous=jnp.linspace(0.0, 1.0, 4).reshape(-1, 1),
        categorical=jnp.array([[0], [1], [0], [1]], dtype=jnp.int32),
    )
    mixed_state = pg.fit_preference_gp(
        mixed_features,
        winner,
        loser,
        dim_continuous=1,
        dim_categorical=1,
        rng=jax.random.PRNGKey(7),
        num_restarts=1,
    )
    mean, std = pg.utility_mean_std(mixed_state, mixed_features)
    self.assertTrue(bool(jnp.all(jnp.isfinite(mean))))
    self.assertTrue(bool(jnp.all(jnp.isfinite(std))))

  def test_thompson_samples_are_finite_and_shaped(self):
    samples = pg.thompson_samples(
        self.state, self.features, num_samples=5, rng=jax.random.PRNGKey(8)
    )
    self.assertEqual(samples.shape, (self.features.num_points, 5))
    self.assertTrue(bool(jnp.all(jnp.isfinite(samples))))

  def test_utility_difference_shapes(self):
    features_a = _continuous_features(self.features.continuous[:4])
    features_b = _continuous_features(self.features.continuous[4:8])
    diff_mean, diff_std = pg.utility_difference(
        self.state, features_a, features_b
    )
    self.assertEqual(diff_mean.shape, (4,))
    self.assertEqual(diff_std.shape, (4,))
    self.assertTrue(bool(jnp.all(jnp.isfinite(diff_mean))))
    self.assertTrue(bool(jnp.all(diff_std >= 0.0)))


if __name__ == '__main__':
  absltest.main()
