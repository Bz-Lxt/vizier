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

from __future__ import annotations

"""Designer learning from pairwise preferences without numeric objectives.

`PreferenceGPDesigner` fits a probit preference Gaussian process (see
`preference_gp.py`) on win/loss comparisons between parameter assignments.
Preferences are supplied either explicitly via `report_preference` or encoded
on completed trials as metadata in the `preference_gp` namespace:

  trial.metadata.ns('preference_gp')['winner_id'] = '<winning trial id>'
  trial.metadata.ns('preference_gp')['loser_id'] = '<losing trial id>'

The designer predicts win probabilities and posterior utility uncertainty for
new parameter pairs and produces de-duplicated batch suggestions via Thompson
sampling. Duplicate parameter assignments, contradictory comparisons, and
incremental feedback are handled by construction.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import attr
import jax
import jax.numpy as jnp
import numpy as np
from vizier import algorithms as vza
from vizier import pyvizier as vz
from vizier._src.algorithms.designers import quasi_random
from vizier._src.algorithms.designers.preference_gp import preference_gp as pg
from vizier.pyvizier import converters

# Metadata namespace and keys used to encode preferences on trials.
PREFERENCE_METADATA_NAMESPACE = 'preference_gp'
WINNER_ID_KEY = 'winner_id'
LOSER_ID_KEY = 'loser_id'

# Number of extra Thompson draws to resolve collisions during batch selection.
_THOMPSON_REDRAWS = 8


@attr.define(frozen=True)
class PreferenceComparison:
  """A single win/loss comparison between two parameter assignments.

  Attributes:
    winner: The preferred parameter assignment.
    loser: The less preferred parameter assignment.
  """

  winner: vz.TrialSuggestion
  loser: vz.TrialSuggestion


def _copy_suggestion(
    suggestion: vz.TrialSuggestion,
) -> vz.TrialSuggestion:
  """Returns a detached copy so later caller mutations do not leak in."""
  return vz.TrialSuggestion(
      parameters=vz.ParameterDict(
          {name: suggestion.parameters[name] for name in suggestion.parameters}
      ),
      metadata=vz.Metadata(dict(suggestion.metadata)),
  )


def _row_key(cont_row: np.ndarray, cat_row: np.ndarray) -> Tuple[bytes, bytes]:
  return (cont_row.tobytes(), cat_row.tobytes())


class PreferenceGPDesigner(vza.Designer, vza.Predictor):
  """Preference Gaussian Process designer.

  Learns a latent utility function from pairwise win/loss comparisons with
  Gaussian evaluation noise, without requiring any numeric objective metric.
  Supports win-probability and posterior-uncertainty prediction for new
  parameter pairs, and de-duplicated batch suggestions through Thompson
  sampling.

  Example:
    designer = PreferenceGPDesigner.from_problem(problem, seed=0)
    designer.report_preference(good_suggestion, bad_suggestion)
    batch = designer.suggest(5)
    probability = designer.predict_win_probability([candidate_a],
                                                   [candidate_b])
  """

  def __init__(
      self,
      problem: vz.ProblemStatement,
      *,
      num_seed_trials: int = 5,
      ard_restarts: int = pg.DEFAULT_RESTARTS,
      newton_iters: int = 15,
      lbfgs_maxiter: int = pg.DEFAULT_LBFGS_MAXITER,
      candidate_pool_size: int = 1000,
      rng: Optional[jax.Array] = None,
      seed: Optional[int] = None,
  ):
    """Initializes the designer.

    Args:
      problem: Flat (non-conditional) problem statement. Metric information is
        optional; preferences do not use numeric objectives.
      num_seed_trials: Number of unique observed parameter points required
        before fitting the GP; earlier suggestions are quasi-random.
      ard_restarts: LBFGS restarts for kernel hyperparameter optimization.
      newton_iters: Newton iterations for each Laplace approximation.
      lbfgs_maxiter: Maximum LBFGS iterations per hyperparameter restart.
      candidate_pool_size: Quasi-random candidate pool size used for Thompson
        batch selection.
      rng: Optional JAX PRNG key. Either `rng` or `seed` may be given.
      seed: Optional integer seed used to create a PRNG key.
    """
    if not jax.config.read('jax_enable_x64'):
      raise ValueError(
          "PreferenceGPDesigner requires JAX x64. Add "
          "jax.config.update('jax_enable_x64', True) before constructing it."
      )
    if problem.search_space.is_conditional:
      raise ValueError(
          'PreferenceGPDesigner does not support conditional search spaces.'
      )
    self._problem = problem
    self._num_seed_trials = num_seed_trials
    self._ard_restarts = ard_restarts
    self._newton_iters = newton_iters
    self._lbfgs_maxiter = lbfgs_maxiter
    self._candidate_pool_size = candidate_pool_size

    if rng is not None and seed is not None:
      raise ValueError('At most one of `rng` and `seed` may be specified.')
    if rng is None:
      rng = jax.random.PRNGKey(seed if seed is not None else 0)
    self._rng = rng

    # No metric information is required; metric sign flipping is disabled.
    self._converter = (
        converters.TrialToModelInputConverter.from_problem(
            problem,
            scale=True,
            max_discrete_indices=0,
            flip_sign_for_minimization_metrics=False,
        )
    )
    self._dim_continuous = (
        self._converter.to_features(
            [vz.TrialSuggestion()]
        ).continuous.padded_array.shape[-1]
    )
    self._dim_categorical = (
        self._converter.to_features(
            [vz.TrialSuggestion()]
        ).categorical.padded_array.shape[-1]
    )
    self._quasi_random = quasi_random.QuasiRandomDesigner(
        problem.search_space, seed=seed
    )

    self._comparisons: List[PreferenceComparison] = []
    self._trial_store: Dict[int, vz.Trial] = {}
    # Lazy fit cache, invalidated whenever new comparisons are incorporated.
    self._cache: Optional[dict] = None

  @classmethod
  def from_problem(
      cls,
      problem: vz.ProblemStatement,
      seed: Optional[int] = None,
      **kwargs,
  ) -> 'PreferenceGPDesigner':
    """Factory matching the `vza.DesignerFactory` protocol."""
    return cls(problem, seed=seed, **kwargs)

  # ---------------------------------------------------------------------------
  # Data ingestion
  # ---------------------------------------------------------------------------
  def report_preference(
      self,
      winner: vz.TrialSuggestion,
      loser: vz.TrialSuggestion,
  ) -> None:
    """Reports that `winner` was preferred over `loser`.

    Repeated comparisons (including identical pairs and contradictory
    directions) are accumulated safely. Incremental calls simply extend the
    internal comparison log.

    Args:
      winner: Preferred parameter assignment (a `TrialSuggestion` or `Trial`).
      loser: Less preferred parameter assignment.
    """
    self._comparisons.append(
        PreferenceComparison(
            winner=_copy_suggestion(winner), loser=_copy_suggestion(loser)
        )
    )
    self._cache = None

  def update(
      self,
      completed: vza.CompletedTrials,
      all_active: vza.ActiveTrials = (),
  ) -> None:
    """Incorporates trials, parsing preferences from trial metadata.

    Completed trials may carry `preference_gp` metadata with `winner_id` and
    `loser_id` entries referencing trials previously seen by this designer
    (via either argument). Trials without such metadata are ignored, so plain
    numeric studies can call `update` without errors.

    Args:
      completed: Newly completed trials.
      all_active: All active trials in the study.

    Raises:
      ValueError: If preference metadata references an unknown trial id or
        only one of the two id keys is present.
    """
    # Active trials are stored first so completed trials take precedence.
    for trial in all_active:
      self._trial_store.setdefault(trial.id, trial)
    for trial in completed:
      self._trial_store[trial.id] = trial

    for trial in completed:
      namespace = trial.metadata.ns(PREFERENCE_METADATA_NAMESPACE)
      winner_id = namespace.get(WINNER_ID_KEY)
      loser_id = namespace.get(LOSER_ID_KEY)
      if winner_id is None and loser_id is None:
        continue
      if winner_id is None or loser_id is None:
        raise ValueError(
            'Preference metadata must define both '
            f'{WINNER_ID_KEY!r} and {LOSER_ID_KEY!r}; got '
            f'winner_id={winner_id!r}, loser_id={loser_id!r} on trial '
            f'{trial.id}.'
        )
      winner = self._resolve_trial(winner_id, trial.id)
      loser = self._resolve_trial(loser_id, trial.id)
      self.report_preference(winner, loser)

  def _resolve_trial(self, raw_trial_id: str, reporting_trial_id: int):
    try:
      trial_id = int(raw_trial_id)
    except (TypeError, ValueError) as exc:
      raise ValueError(
          f'Preference metadata on trial {reporting_trial_id} references a '
          f'non-integer trial id {raw_trial_id!r}.'
      ) from exc
    if trial_id not in self._trial_store:
      raise ValueError(
          f'Preference metadata on trial {reporting_trial_id} references '
          f'unknown trial id {trial_id}. Ensure the referenced trial was '
          'previously passed to update().'
      )
    return self._trial_store[trial_id]

  # ---------------------------------------------------------------------------
  # Feature engineering and fitting
  # ---------------------------------------------------------------------------
  def _to_feature_rows(
      self, suggestions: Sequence[vz.TrialSuggestion]
  ) -> Tuple[np.ndarray, np.ndarray]:
    model_input = self._converter.to_features(list(suggestions))
    continuous = np.asarray(
        model_input.continuous.padded_array, dtype=np.float64
    )
    categorical = np.asarray(
        model_input.categorical.padded_array, dtype=np.int32
    )
    return continuous, categorical

  def _build_training_arrays(
      self,
  ) -> Tuple[pg.PreferenceFeatures, np.ndarray, np.ndarray, set]:
    """Deduplicates parameter points and maps comparisons to unique indices."""
    key_to_index: Dict[Tuple[bytes, bytes], int] = {}
    continuous_rows: List[np.ndarray] = []
    categorical_rows: List[np.ndarray] = []
    winner_idx: List[int] = []
    loser_idx: List[int] = []

    def index_of(suggestion: vz.TrialSuggestion) -> int:
      continuous, categorical = self._to_feature_rows([suggestion])
      key = _row_key(continuous[0], categorical[0])
      existing = key_to_index.get(key)
      if existing is None:
        existing = len(continuous_rows)
        key_to_index[key] = existing
        continuous_rows.append(continuous[0])
        categorical_rows.append(categorical[0])
      return existing

    for comparison in self._comparisons:
      winner_idx.append(index_of(comparison.winner))
      loser_idx.append(index_of(comparison.loser))

    if continuous_rows:
      continuous = np.stack(continuous_rows, axis=0).astype(np.float64)
      categorical = np.stack(categorical_rows, axis=0).astype(np.int32)
    else:
      continuous = np.zeros(
          (0, self._dim_continuous), dtype=np.float64
      )
      categorical = np.zeros(
          (0, self._dim_categorical), dtype=np.int32
      )
    features = pg.PreferenceFeatures(
        continuous=continuous,
        categorical=categorical,
    )
    return (
        features,
        np.asarray(winner_idx, dtype=np.int32),
        np.asarray(loser_idx, dtype=np.int32),
        set(key_to_index.keys()),
    )

  def _fit(self) -> dict:
    """Fits (or reuses the cached fit) and returns the fit cache."""
    comparison_count = len(self._comparisons)
    if self._cache is not None and (
        self._cache['comparison_count'] == comparison_count
    ):
      return self._cache

    features, winner_idx, loser_idx, observed_keys = (
        self._build_training_arrays()
    )
    self._rng, fit_rng = jax.random.split(self._rng)
    state = pg.fit_preference_gp(
        features,
        jnp.asarray(winner_idx),
        jnp.asarray(loser_idx),
        dim_continuous=self._dim_continuous,
        dim_categorical=self._dim_categorical,
        rng=fit_rng,
        num_restarts=self._ard_restarts,
        num_newton_iters=self._newton_iters,
        lbfgs_maxiter=self._lbfgs_maxiter,
    )
    self._cache = {
        'comparison_count': comparison_count,
        'state': state,
        'features': features,
        'observed_keys': observed_keys,
    }
    return self._cache

  @property
  def num_comparisons(self) -> int:
    """Number of preference comparisons accumulated so far."""
    return len(self._comparisons)

  # ---------------------------------------------------------------------------
  # Prediction
  # ---------------------------------------------------------------------------
  def _features_of(
      self, suggestions: Sequence[vz.TrialSuggestion]
  ) -> pg.PreferenceFeatures:
    continuous, categorical = self._to_feature_rows(suggestions)
    return pg.PreferenceFeatures(
        continuous=jnp.asarray(continuous),
        categorical=jnp.asarray(categorical),
    )

  def predict(
      self,
      trials: Sequence[vz.TrialSuggestion],
      rng: Optional[jax.Array] = None,
      num_samples: Optional[int] = None,
  ) -> vza.Prediction:
    """Returns posterior latent utility mean and standard deviation."""
    del rng, num_samples  # The Laplace posterior is deterministic.
    cache = self._fit()
    mean, std = pg.utility_mean_std(
        cache['state'], self._features_of(trials)
    )
    return vza.Prediction(
        mean=np.asarray(mean, dtype=np.float64),
        stddev=np.asarray(std, dtype=np.float64),
    )

  def predict_utility(
      self, trials: Sequence[vz.TrialSuggestion]
  ) -> vza.Prediction:
    """Alias for `predict`: posterior utility mean and standard deviation."""
    return self.predict(trials)

  def predict_win_probability(
      self,
      trials_a: Sequence[vz.TrialSuggestion],
      trials_b: Sequence[vz.TrialSuggestion],
  ) -> np.ndarray:
    """Returns aligned win probabilities ``P(a beats b)`` in ``(0, 1)``."""
    cache = self._fit()
    probabilities = pg.win_probability(
        cache['state'],
        self._features_of(trials_a),
        self._features_of(trials_b),
    )
    return np.asarray(probabilities, dtype=np.float64)

  def predict_utility_difference(
      self,
      trials_a: Sequence[vz.TrialSuggestion],
      trials_b: Sequence[vz.TrialSuggestion],
  ) -> vza.Prediction:
    """Returns posterior mean and stddev of the utility difference ``f(a)-f(b)``."""
    cache = self._fit()
    diff_mean, diff_std = pg.utility_difference(
        cache['state'],
        self._features_of(trials_a),
        self._features_of(trials_b),
    )
    return vza.Prediction(
        mean=np.asarray(diff_mean, dtype=np.float64),
        stddev=np.asarray(diff_std, dtype=np.float64),
    )

  # ---------------------------------------------------------------------------
  # Suggestion
  # ---------------------------------------------------------------------------
  def _quasi_random_unseen(
      self, count: int, observed_keys: set
  ) -> List[vz.TrialSuggestion]:
    """Draws quasi-random suggestions whose parameters were not observed."""
    suggestions: List[vz.TrialSuggestion] = []
    seen_keys = set(observed_keys)
    pool_multiplier = 1
    while len(suggestions) < count:
      pool = self._quasi_random.suggest(count * pool_multiplier)
      for candidate in pool:
        continuous, categorical = self._to_feature_rows([candidate])
        key = _row_key(continuous[0], categorical[0])
        if key in seen_keys:
          continue
        seen_keys.add(key)
        suggestions.append(candidate)
        if len(suggestions) == count:
          break
      pool_multiplier += 1
      if pool_multiplier > 16:
        # Extremely small discrete spaces can exhaust unseen points; fall
        # back to the raw pool rather than looping forever.
        suggestions.extend(pool)
        break
    return suggestions[:count]

  def suggest(self, count: int = 1) -> Sequence[vz.TrialSuggestion]:
    """Returns `count` de-duplicated, valid parameter suggestions.

    Before `num_seed_trials` unique points have been observed, suggestions are
    quasi-random. Afterwards, a quasi-random candidate pool is scored with the
    fitted preference GP and batch members are selected via Thompson sampling
    with posterior re-draws to avoid duplicates.
    """
    if count <= 0:
      raise ValueError(f'`count` must be positive, got {count}.')
    cache = self._fit()
    observed_keys = set(cache['observed_keys'])

    if len(observed_keys) < self._num_seed_trials:
      return self._quasi_random_unseen(count, observed_keys)

    pool = self._quasi_random.suggest(self._candidate_pool_size)
    pool_continuous, pool_categorical = self._to_feature_rows(pool)
    # Deduplicate the pool itself and exclude already observed parameters.
    unique_pool: List[vz.TrialSuggestion] = []
    pool_keys: set = set()
    for index, candidate in enumerate(pool):
      key = _row_key(pool_continuous[index], pool_categorical[index])
      if key in observed_keys or key in pool_keys:
        continue
      pool_keys.add(key)
      unique_pool.append(candidate)
    if not unique_pool:
      return self._quasi_random_unseen(count, observed_keys)

    pool_features = self._features_of(unique_pool)
    state = cache['state']
    self._rng, thompson_rng = jax.random.split(self._rng)
    samples = pg.thompson_samples(
        state,
        pool_features,
        num_samples=count + _THOMPSON_REDRAWS,
        rng=thompson_rng,
    )
    samples = np.asarray(samples)
    posterior_mean, _ = pg.utility_mean_std(state, pool_features)
    posterior_mean = np.asarray(posterior_mean)

    chosen_indices: List[int] = []
    chosen_keys: set = set()
    for draw in range(samples.shape[1]):
      order = np.argsort(-samples[:, draw])
      for pool_index in order:
        key = _row_key(
            pool_continuous[pool_index], pool_categorical[pool_index]
        )
        if key not in chosen_keys:
          chosen_keys.add(key)
          chosen_indices.append(int(pool_index))
          break
      if len(chosen_indices) == count:
        break

    # Fallback: fill any remaining slots by posterior mean ranking.
    if len(chosen_indices) < count:
      for pool_index in np.argsort(-posterior_mean):
        pool_index = int(pool_index)
        key = _row_key(
            pool_continuous[pool_index], pool_categorical[pool_index]
        )
        if key not in chosen_keys:
          chosen_keys.add(key)
          chosen_indices.append(pool_index)
        if len(chosen_indices) == count:
          break

    # Last-resort fallback for tiny discrete spaces.
    if len(chosen_indices) < count:
      extra = self._quasi_random_unseen(
          count - len(chosen_indices), observed_keys | chosen_keys
      )
      chosen_suggestions = [unique_pool[i] for i in chosen_indices]
      chosen_suggestions.extend(extra)
    else:
      chosen_suggestions = [unique_pool[i] for i in chosen_indices]

    suggestions: List[vz.TrialSuggestion] = []
    for candidate in chosen_suggestions[:count]:
      self._problem.search_space.assert_contains(candidate.parameters)
      metadata = vz.Metadata(dict(candidate.metadata))
      metadata.ns(PREFERENCE_METADATA_NAMESPACE)['model'] = (
          'preference_gp'
      )
      metadata.ns(PREFERENCE_METADATA_NAMESPACE)['acquisition'] = (
          'thompson_sampling'
      )
      suggestions.append(
          vz.TrialSuggestion(
              parameters=vz.ParameterDict(
                  {
                      name: candidate.parameters[name]
                      for name in candidate.parameters
                  }
              ),
              metadata=metadata,
          )
      )
    return suggestions
