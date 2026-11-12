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

import numpy as np
from vizier import pyvizier as vz
from vizier._src.algorithms.designers.preference_gp import (
    preference_gp_designer as pgd,
)
from vizier._src.algorithms.designers.preference_gp.preference_gp_designer import (
    PREFERENCE_METADATA_NAMESPACE,
    LOSER_ID_KEY,
    PreferenceComparison,
    PreferenceGPDesigner,
    WINNER_ID_KEY,
)


def _continuous_problem() -> vz.ProblemStatement:
  search_space = vz.SearchSpace()
  search_space.root.add_float_param('x', 0.0, 1.0)
  # Deliberately no MetricInformation: preferences need no numeric objective.
  return vz.ProblemStatement(search_space=search_space)


def _mixed_problem() -> vz.ProblemStatement:
  search_space = vz.SearchSpace()
  search_space.root.add_float_param('x', -1.0, 1.0)
  search_space.root.add_int_param('k', 1, 4)
  search_space.root.add_categorical_param('c', ['a', 'b', 'c'])
  return vz.ProblemStatement(search_space=search_space)


def _noisy_preference_oracle(
    designer: PreferenceGPDesigner,
    seen,
    utility_fn,
    rng: np.random.Generator,
    num_comparisons: int,
    noise_sd: float = 0.05,
) -> None:
  for _ in range(num_comparisons):
    i, j = rng.integers(0, len(seen), size=2)
    if i == j:
      continue
    winner, loser = seen[i], seen[j]
    utility_winner = utility_fn(winner) + rng.normal(0.0, noise_sd)
    utility_loser = utility_fn(loser) + rng.normal(0.0, noise_sd)
    if utility_winner > utility_loser:
      designer.report_preference(winner, loser)
    else:
      designer.report_preference(loser, winner)


class PreferenceGPDesignerTest(parameterized.TestCase):

  def test_public_api_and_construction_without_metrics(self):
    designer = PreferenceGPDesigner.from_problem(
        _continuous_problem(), seed=0, ard_restarts=1
    )
    for method in [
        'report_preference',
        'update',
        'suggest',
        'predict',
        'predict_utility',
        'predict_win_probability',
        'predict_utility_difference',
    ]:
      self.assertTrue(hasattr(designer, method))
    self.assertIsInstance(designer, PreferenceGPDesigner)

  def test_construction_requires_x64(self):
    jax.config.update('jax_enable_x64', False)
    try:
      with self.assertRaises(ValueError):
        PreferenceGPDesigner(
            _continuous_problem(), seed=0, ard_restarts=1
        )
    finally:
      jax.config.update('jax_enable_x64', True)

  def test_seed_phase_and_empty_prior(self):
    designer = PreferenceGPDesigner(
        _continuous_problem(),
        seed=0,
        num_seed_trials=5,
        ard_restarts=1,
    )
    suggestions = designer.suggest(3)
    self.assertLen(suggestions, 3)
    keys = {
        s.parameters['x'].value
        for s in suggestions
    }
    self.assertLen(keys, 3)
    prediction = designer.predict_utility(suggestions)
    np.testing.assert_allclose(prediction.mean, np.zeros(3), atol=1e-10)
    self.assertTrue(np.all(np.isfinite(prediction.stddev)))
    probabilities = designer.predict_win_probability(
        suggestions[:2], suggestions[1:3]
    )
    np.testing.assert_allclose(probabilities, np.full(2, 0.5), atol=1e-10)

  def test_batch_suggestions_valid_and_deduplicated(self):
    problem = _continuous_problem()
    designer = PreferenceGPDesigner(
        problem, seed=1, num_seed_trials=3, ard_restarts=1,
        candidate_pool_size=200,
    )
    seen = designer.suggest(6)
    rng = np.random.default_rng(0)
    _noisy_preference_oracle(
        designer,
        seen,
        lambda s: -abs(s.parameters['x'].value - 0.4),
        rng,
        num_comparisons=30,
    )
    batch = designer.suggest(5)
    self.assertLen(batch, 5)
    values = [s.parameters['x'].value for s in batch]
    self.assertLen(set(values), 5)
    for suggestion in batch:
      problem.search_space.assert_contains(suggestion.parameters)
    observed = designer._fit()['observed_keys']  # pylint: disable=protected-access
    continuous, categorical = designer._to_feature_rows(batch)  # pylint: disable=protected-access
    for index in range(5):
      key = (
          continuous[index].tobytes(),
          categorical[index].tobytes(),
      )
      self.assertNotIn(key, observed)

  def test_duplicate_parameter_assignments_share_unique_rows(self):
    designer = PreferenceGPDesigner(
        _continuous_problem(), seed=2, num_seed_trials=1, ard_restarts=1
    )
    first = vz.TrialSuggestion({'x': 0.3})
    second = vz.TrialSuggestion({'x': 0.7})
    duplicate = vz.TrialSuggestion({'x': 0.3})
    for _ in range(5):
      designer.report_preference(first, second)
    designer.report_preference(second, duplicate)
    cache = designer._fit()  # pylint: disable=protected-access
    self.assertEqual(cache['state'].num_points, 2)
    self.assertEqual(designer.num_comparisons, 6)
    prediction = designer.predict_utility([first, second, duplicate])
    self.assertTrue(np.all(np.isfinite(prediction.mean)))

  def test_contradictory_comparisons_remain_finite(self):
    designer = PreferenceGPDesigner(
        _continuous_problem(), seed=3, num_seed_trials=1, ard_restarts=1
    )
    first = vz.TrialSuggestion({'x': 0.2})
    second = vz.TrialSuggestion({'x': 0.8})
    for _ in range(4):
      designer.report_preference(first, second)
      designer.report_preference(second, first)
    probability = designer.predict_win_probability([first], [second])
    self.assertTrue(np.isfinite(probability).all())
    self.assertAlmostEqual(float(probability[0]), 0.5, delta=0.1)
    suggestions = designer.suggest(3)
    self.assertLen(suggestions, 3)

  def test_incremental_feedback_equivalent_and_cache(self):
    problem = _continuous_problem()
    one_shot = PreferenceGPDesigner(
        problem, seed=4, num_seed_trials=2, ard_restarts=1
    )
    incremental = PreferenceGPDesigner(
        problem, seed=4, num_seed_trials=2, ard_restarts=1
    )
    rng = np.random.default_rng(4)
    points = [vz.TrialSuggestion({'x': float(v)})
              for v in np.linspace(0.1, 0.9, 6)]
    comparisons = []
    for _ in range(24):
      i, j = rng.integers(0, 6, size=2)
      if i == j:
        continue
      if rng.random() > 0.5:
        comparisons.append(PreferenceComparison(points[i], points[j]))
      else:
        comparisons.append(PreferenceComparison(points[j], points[i]))

    for comparison in comparisons:
      one_shot.report_preference(
          comparison.winner, comparison.loser
      )
    chunks = np.array_split(np.arange(len(comparisons)), 3)
    for chunk in chunks:
      for index in chunk:
        comparison = comparisons[index]
        incremental.report_preference(
            comparison.winner, comparison.loser
        )

    queries = [vz.TrialSuggestion({'x': v}) for v in (0.2, 0.5, 0.8)]
    probability_one_shot = one_shot.predict_win_probability(
        queries[:2], queries[1:3]
    )
    probability_incremental = incremental.predict_win_probability(
        queries[:2], queries[1:3]
    )
    np.testing.assert_allclose(
        probability_one_shot, probability_incremental, atol=1e-6
    )

    # Cache is reused without new data and invalidated on new comparisons.
    cached = incremental._fit()  # pylint: disable=protected-access
    self.assertIs(incremental._fit(), cached)  # pylint: disable=protected-access
    incremental.report_preference(points[0], points[1])
    self.assertIsNone(incremental._cache)  # pylint: disable=protected-access

  def test_update_parses_preference_metadata(self):
    designer = PreferenceGPDesigner(
        _continuous_problem(), seed=5, num_seed_trials=1, ard_restarts=1
    )
    winner = vz.Trial(
        id=1, parameters=vz.ParameterDict({'x': vz.ParameterValue(0.8)})
    )
    loser = vz.Trial(
        id=2, parameters=vz.ParameterDict({'x': vz.ParameterValue(0.2)})
    )
    reporter = vz.Trial(id=3)
    reporter.metadata.ns(PREFERENCE_METADATA_NAMESPACE)[WINNER_ID_KEY] = '1'
    reporter.metadata.ns(PREFERENCE_METADATA_NAMESPACE)[LOSER_ID_KEY] = '2'
    designer.update(completed=[reporter], all_active=[winner, loser])
    self.assertEqual(designer.num_comparisons, 1)
    probability = designer.predict_win_probability(
        [vz.TrialSuggestion({'x': 0.8})],
        [vz.TrialSuggestion({'x': 0.2})],
    )
    self.assertGreater(float(probability[0]), 0.5)

  def test_update_ignores_numeric_trials_and_rejects_bad_metadata(self):
    designer = PreferenceGPDesigner(
        _continuous_problem(), seed=6, num_seed_trials=1, ard_restarts=1
    )
    numeric_trial = vz.Trial(id=4)
    numeric_trial.final_measurement = vz.Measurement(
        metrics={'objective': 1.0}
    )
    designer.update(completed=[numeric_trial])
    self.assertEqual(designer.num_comparisons, 0)

    unknown_id_trial = vz.Trial(id=5)
    unknown_id_trial.metadata.ns(PREFERENCE_METADATA_NAMESPACE)[
        WINNER_ID_KEY
    ] = '99'
    unknown_id_trial.metadata.ns(PREFERENCE_METADATA_NAMESPACE)[
        LOSER_ID_KEY
    ] = '4'
    with self.assertRaises(ValueError):
      designer.update(completed=[unknown_id_trial])

    partial_trial = vz.Trial(id=6)
    partial_trial.metadata.ns(PREFERENCE_METADATA_NAMESPACE)[
        WINNER_ID_KEY
    ] = '4'
    with self.assertRaises(ValueError):
      designer.update(completed=[partial_trial])

  def test_mixed_space_end_to_end(self):
    problem = _mixed_problem()
    designer = PreferenceGPDesigner(
        problem, seed=7, num_seed_trials=3, ard_restarts=1,
        candidate_pool_size=120,
    )
    rng = np.random.default_rng(7)
    seen = []
    for _ in range(4):
      batch = designer.suggest(4)
      for suggestion in batch:
        problem.search_space.assert_contains(suggestion.parameters)
      seen.extend(batch)
      _noisy_preference_oracle(
          designer,
          seen,
          lambda s: -abs(s.parameters['x'].value),
          rng,
          num_comparisons=16,
      )
    final = designer.suggest(5)
    self.assertLen(final, 5)
    unique = {
        tuple(sorted(suggestion.parameters.as_dict().items()))
        for suggestion in final
    }
    self.assertLen(unique, 5)
    prediction = designer.predict_utility(final)
    self.assertEqual(prediction.mean.shape, (5,))
    self.assertEqual(prediction.stddev.shape, (5,))
    self.assertTrue(np.all(np.isfinite(prediction.mean)))
    probabilities = designer.predict_win_probability(
        final[:2], final[3:5]
    )
    self.assertTrue(np.all(probabilities > 0.0))
    self.assertTrue(np.all(probabilities < 1.0))
    difference = designer.predict_utility_difference(
        final[:2], final[3:5]
    )
    self.assertTrue(np.all(np.isfinite(difference.stddev)))

  def test_end_to_end_preference_optimization_outperforms_random(self):
    optimum = 0.35

    def utility(x: float) -> float:
      return -((x - optimum) ** 2)

    problem = _continuous_problem()
    designer = PreferenceGPDesigner(
        problem, seed=0, num_seed_trials=4, ard_restarts=1,
        candidate_pool_size=300,
    )
    rng = np.random.default_rng(0)
    seen = []
    for _ in range(8):
      seen.extend(designer.suggest(4))
      _noisy_preference_oracle(
          designer,
          seen,
          lambda s: utility(s.parameters['x'].value),
          rng,
          num_comparisons=12,
      )
    final = designer.suggest(8)
    designer_x = np.array(
        [suggestion.parameters['x'].value for suggestion in final]
    )
    designer_utility = utility(designer_x)

    baseline_rng = np.random.default_rng(1000)
    random_utility = utility(baseline_rng.uniform(0.0, 1.0, size=8))

    # The designer should concentrate suggestions near the optimum and clearly
    # beat the uniform-random baseline on average utility.
    self.assertGreater(
        float(designer_utility.mean()), float(random_utility.mean())
    )
    best_distance = float(
        np.abs(designer_x[np.argmax(designer_utility)] - optimum)
    )
    self.assertLess(best_distance, 0.1)
    self.assertGreater(float(designer_utility.max()), -0.01)


if __name__ == '__main__':
  absltest.main()
