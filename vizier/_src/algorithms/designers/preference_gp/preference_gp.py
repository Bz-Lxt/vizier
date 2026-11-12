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

"""Preference Gaussian Process math core.

A latent utility function ``f ~ GP(0, k)`` is learned from pairwise win/loss
comparisons only.  Each comparison assumes Gaussian evaluation noise: a winner
``w`` and loser ``l`` satisfy

  ``P(w beats l | f) = Phi((f(w) - f(l)) / s)`` with ``s = sqrt(2 * sigma_n^2)``

where ``sigma_n^2`` is the per-evaluation Gaussian noise variance and ``Phi`` is
the standard normal CDF (probit link).

The posterior over the latent utilities at the observed points is approximated
with Laplace's method (Newton iterations + Cholesky solves). Kernel
hyperparameters (signal variance, ARD length scales, noise variance) are fit by
maximizing the Laplace log marginal likelihood with multi-restart LBFGS.

The module is intentionally free of Vizier Trial types: everything operates on
continuous/categorical feature arrays and integer comparison index pairs.
"""

import functools
from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxopt
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfpk = tfp.math.psd_kernels
tfpke = tfp.experimental.psd_kernels
tfp_marginal = tfp.experimental.distributions.marginal_fns

# Default numerical settings.
DEFAULT_NEWTON_ITERS = 20
DEFAULT_RESTARTS = 4
DEFAULT_LBFGS_MAXITER = 50
# Jitter added to kernel matrices and posterior covariances.
_JITTER = 1e-6
# Initial jitter for the retrying Cholesky of ``I + W^{1/2} T K T' W^{1/2}``.
_CHOL_JITTER = 1e-4
_CHOL_MAX_ITERS = 5
# Below this standard-normal argument the inverse Mills ratio is approximated by
# its asymptotic form to avoid (extreme) underflow.
_INVERSE_MILLS_THRESHOLD = -35.0
_DEFAULT_INITIAL_NOISE_VARIANCE = 1e-2
# Weak log-normal hyperpriors prevent the type-II marginal likelihood from
# collapsing onto the trivial (vanishing signal variance) solution when only a
# handful of comparisons are available. Features are scaled to O(1), so length
# scales near 1 and variances near the defaults are sensible prior modes.
_HYPERPRIOR_STD = 1.5
_LENGTH_SCALE_PRIOR_MEAN = 0.0
_SIGNAL_VARIANCE_PRIOR_MEAN = 0.0

_normal_logcdf = jax.scipy.stats.norm.logcdf
_normal_logpdf = jax.scipy.stats.norm.logpdf
_normal_cdf = jax.scipy.stats.norm.cdf


class PreferenceFeatures(eqx.Module):
  """Feature arrays for a collection of (possibly deduplicated) parameter points.

  Attributes:
    continuous: Float array of shape ``(num_points, num_continuous_features)``;
      may have zero features per point.
    categorical: Integer array of shape
      ``(num_points, num_categorical_features)``; may have zero features.
  """

  continuous: jax.Array
  categorical: jax.Array

  @property
  def num_points(self) -> int:
    return self.continuous.shape[0]

  @classmethod
  def empty(cls, dim_continuous: int, dim_categorical: int) -> 'PreferenceFeatures':
    return cls(
        continuous=jnp.zeros((0, dim_continuous), dtype=jnp.float64),
        categorical=jnp.zeros((0, dim_categorical), dtype=jnp.int32),
    )

  def concat(self, other: 'PreferenceFeatures') -> 'PreferenceFeatures':
    return PreferenceFeatures(
        continuous=jnp.concatenate([self.continuous, other.continuous], axis=0),
        categorical=jnp.concatenate(
            [self.categorical, other.categorical], axis=0
        ),
    )


class PreferenceGPParams(eqx.Module):
  """Trainable hyperparameters, all stored in unconstrained log space.

  Attributes:
    log_signal_variance: Scalar log of the kernel signal variance.
    log_continuous_length_scale: Log ARD length scales, shape
      ``(dim_continuous,)``; may be empty.
    log_categorical_length_scale: Log ARD length scales, shape
      ``(dim_categorical,)``; may be empty.
    log_noise_variance: Scalar log of the per-evaluation Gaussian noise
      variance ``sigma_n^2``.
  """

  log_signal_variance: jax.Array
  log_continuous_length_scale: jax.Array
  log_categorical_length_scale: jax.Array
  log_noise_variance: jax.Array

  @classmethod
  def initial(
      cls,
      dim_continuous: int,
      dim_categorical: int,
      *,
      rng: Optional[jax.Array] = None,
      perturbation: float = 0.0,
  ) -> 'PreferenceGPParams':
    """Builds an initial parameter set, optionally log-space perturbed."""
    if rng is None:
      rng = jax.random.PRNGKey(0)
    keys = jax.random.split(rng, 4)
    return cls(
        log_signal_variance=jax.random.normal(keys[0]) * perturbation,
        log_continuous_length_scale=(
            jax.random.normal(keys[1], shape=(dim_continuous,)) * perturbation
        ),
        log_categorical_length_scale=(
            jax.random.normal(keys[2], shape=(dim_categorical,)) * perturbation
        ),
        log_noise_variance=(
            jnp.log(_DEFAULT_INITIAL_NOISE_VARIANCE)
            + jax.random.normal(keys[3]) * perturbation
        ),
    )

  @property
  def noise_variance(self) -> jax.Array:
    return jnp.exp(self.log_noise_variance)

  @property
  def comparison_scale(self) -> jax.Array:
    """Standard deviation of the noisy utility difference, ``sqrt(2 sigma_n^2)``."""
    return jnp.sqrt(2.0 * self.noise_variance)


class PreferenceGPState(eqx.Module):
  """Trained preference GP state sufficient for prediction.

  Attributes:
    params: Learned hyperparameters (log space).
    features: Unique training points.
    winner_idx: Winner index per comparison, shape ``(num_comparisons,)``.
    loser_idx: Loser index per comparison, shape ``(num_comparisons,)``.
    f_mode: Laplace posterior mode of the latent utility, shape
      ``(num_points,)``.
    gram: Training kernel matrix (with jitter), shape ``(num_points, num_points)``.
    alpha: ``gram^{-1} f_mode``.
    cholesky: Lower Cholesky factor of
      ``I + W^{1/2} T K T' W^{1/2}``, shape ``(m, m)``.
    sqrt_w: Square root of the negative probit Hessian diagonal, shape ``(m,)``.
    w: Negative probit Hessian diagonal, shape ``(m,)``.
    dim_continuous: Number of continuous feature dimensions (static).
    dim_categorical: Number of categorical feature dimensions (static).
  """

  params: PreferenceGPParams
  features: PreferenceFeatures
  winner_idx: jax.Array
  loser_idx: jax.Array
  f_mode: jax.Array
  gram: jax.Array
  alpha: jax.Array
  cholesky: jax.Array
  sqrt_w: jax.Array
  w: jax.Array
  dim_continuous: int = eqx.field(static=True)
  dim_categorical: int = eqx.field(static=True)

  @property
  def num_comparisons(self) -> int:
    return self.winner_idx.shape[0]

  @property
  def num_points(self) -> int:
    return self.features.num_points


def _index_points(
    features: PreferenceFeatures,
    dim_continuous: int,
    dim_categorical: int,
):
  """Wraps features into the index-point structure expected by the kernel."""
  if dim_categorical == 0:
    # A continuous-only kernel expects plain arrays rather than the
    # continuous/categorical container.
    return features.continuous
  return tfpke.ContinuousAndCategoricalValues(
      continuous=features.continuous, categorical=features.categorical
  )


def build_kernel(
    params: PreferenceGPParams,
    dim_continuous: int,
    dim_categorical: int,
) -> tfpk.PositiveSemidefiniteKernel:
  """Builds the ARD Matern 5/2 kernel described by `params`."""
  amplitude = jnp.sqrt(jnp.exp(params.log_signal_variance))
  base_kernel = tfpk.MaternFiveHalves(amplitude=amplitude)
  continuous_ls = jnp.exp(params.log_continuous_length_scale)
  categorical_ls = jnp.exp(params.log_categorical_length_scale)
  if dim_continuous > 0 and dim_categorical > 0:
    return tfpke.FeatureScaledWithCategorical(
        base_kernel,
        scale_diag=tfpke.ContinuousAndCategoricalValues(
            continuous=continuous_ls, categorical=categorical_ls
        ),
    )
  elif dim_continuous > 0:
    return tfpk.FeatureScaled(base_kernel, scale_diag=continuous_ls)
  else:
    # Categorical-only space: empty continuous index points are valid for the
    # mixed kernel, which reduces to the categorical feature-scaled kernel.
    return tfpke.FeatureScaledWithCategorical(
        base_kernel,
        scale_diag=tfpke.ContinuousAndCategoricalValues(
            continuous=jnp.zeros((0,), dtype=jnp.float64),
            categorical=categorical_ls,
        ),
    )


def _kernel_matrix(
    kernel: tfpk.PositiveSemidefiniteKernel,
    features_a: PreferenceFeatures,
    features_b: PreferenceFeatures,
    dim_continuous: int,
    dim_categorical: int,
) -> jax.Array:
  return kernel.matrix(
      _index_points(features_a, dim_continuous, dim_categorical),
      _index_points(features_b, dim_continuous, dim_categorical),
  )


def _stable_inverse_mills(u: jax.Array) -> jax.Array:
  """Computes ``phi(u) / Phi(u)`` (inverse Mills ratio) stably.

  For very negative ``u`` the direct exponential can underflow; the asymptotic
  Mills approximation ``r(u) ~ -u`` is used there instead.
  """
  ratio = jnp.exp(_normal_logpdf(u) - _normal_logcdf(u))
  return jnp.where(u > _INVERSE_MILLS_THRESHOLD, ratio, -u)


def _retrying_cholesky(matrix: jax.Array) -> jax.Array:
  return tfp_marginal.retrying_cholesky(
      matrix, jitter=_CHOL_JITTER, max_iters=_CHOL_MAX_ITERS
  )[0]


def _laplace_mode(
    params: PreferenceGPParams,
    features: PreferenceFeatures,
    winner_idx: jax.Array,
    loser_idx: jax.Array,
    dim_continuous: int,
    dim_categorical: int,
    num_newton_iters: int,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Runs fixed Newton iterations for the Laplace posterior mode.

  Returns ``(f_mode, cholesky, sqrt_w, w, gram)``.
  """
  kernel = build_kernel(params, dim_continuous, dim_categorical)
  num_points = features.num_points
  gram = _kernel_matrix(
      kernel, features, features, dim_continuous, dim_categorical
  )
  gram = gram + _JITTER * jnp.eye(num_points, dtype=gram.dtype)
  scale = params.comparison_scale

  def newton_body(_, f):
    z = f[winner_idx] - f[loser_idx]
    u = z / scale
    inverse_mills = _stable_inverse_mills(u)
    # Gradient of the probit log likelihood w.r.t. each difference z.
    grad_loglik = inverse_mills / scale
    # Negative second derivative (diagonal); clipped for numerical safety.
    w = jnp.maximum(
        inverse_mills * (u + inverse_mills) / jnp.square(scale), 0.0
    )
    sqrt_w = jnp.sqrt(w)
    # ``T K`` where row k of T is ``e_winner - e_loser``.
    tk = gram[winner_idx] - gram[loser_idx]
    # ``T K T'`` computed with index gathers so T never needs to be materialized.
    tktt = tk[:, winner_idx] - tk[:, loser_idx]
    num_comparisons = winner_idx.shape[0]
    bord = jnp.eye(num_comparisons, dtype=gram.dtype) + (
        sqrt_w[:, jnp.newaxis] * tktt * sqrt_w[jnp.newaxis, :]
    )
    cholesky = _retrying_cholesky(bord)
    # Exact Newton step via Woodbury, with B = I + W^{1/2} T K T' W^{1/2}:
    # f = K T' [g - W^{1/2} B^{-1} W^{1/2} (T K T' g - T f)].
    tkt_g = tktt @ grad_loglik
    solved = jax.scipy.linalg.solve_triangular(
        cholesky, sqrt_w * (tkt_g - z), lower=True
    )
    solved = jax.scipy.linalg.solve_triangular(
        cholesky.T, solved, lower=False
    )
    rhs = grad_loglik - sqrt_w * solved
    t_rhs = (
        jnp.zeros(num_points, dtype=gram.dtype)
        .at[winner_idx]
        .add(rhs)
        .at[loser_idx]
        .add(-rhs)
    )
    return gram @ t_rhs

  f_mode = jax.lax.fori_loop(
      0, num_newton_iters, newton_body,
      jnp.zeros(num_points, dtype=gram.dtype),
  )

  # Recompute the final likelihood curvature and Cholesky for prediction.
  z = f_mode[winner_idx] - f_mode[loser_idx]
  u = z / scale
  inverse_mills = _stable_inverse_mills(u)
  w = jnp.maximum(
      inverse_mills * (u + inverse_mills) / jnp.square(scale), 0.0
  )
  sqrt_w = jnp.sqrt(w)
  tk = gram[winner_idx] - gram[loser_idx]
  tktt = tk[:, winner_idx] - tk[:, loser_idx]
  num_comparisons = winner_idx.shape[0]
  bord = jnp.eye(num_comparisons, dtype=gram.dtype) + (
      sqrt_w[:, jnp.newaxis] * tktt * sqrt_w[jnp.newaxis, :]
  )
  cholesky = _retrying_cholesky(bord)
  return f_mode, cholesky, sqrt_w, w, gram


def _log_hyperprior(params: PreferenceGPParams) -> jax.Array:
  """Log density of the weak log-normal hyperpriors over hyperparameters."""
  noise_prior_mean = jnp.log(_DEFAULT_INITIAL_NOISE_VARIANCE)

  def _log_normal_term(value: jax.Array, mean: float) -> jax.Array:
    return jnp.sum(
        _normal_logpdf(
            (value - mean) / _HYPERPRIOR_STD
        )
        - jnp.log(_HYPERPRIOR_STD)
    )

  return (
      _log_normal_term(
          params.log_signal_variance, _SIGNAL_VARIANCE_PRIOR_MEAN
      )
      + _log_normal_term(
          params.log_continuous_length_scale,
          _LENGTH_SCALE_PRIOR_MEAN,
      )
      + _log_normal_term(
          params.log_categorical_length_scale,
          _LENGTH_SCALE_PRIOR_MEAN,
      )
      + _log_normal_term(params.log_noise_variance, noise_prior_mean)
  )


def _neg_log_marginal_likelihood(
    params: PreferenceGPParams,
    features: PreferenceFeatures,
    winner_idx: jax.Array,
    loser_idx: jax.Array,
    dim_continuous: int,
    dim_categorical: int,
    num_newton_iters: int,
) -> jax.Array:
  """Type-II training objective.

  Returns the negative Laplace log marginal likelihood minus the log
  hyperprior density, i.e. the maximum-a-posteriori objective used for
  hyperparameter optimization.
  """
  f_mode, cholesky, _, _, gram = _laplace_mode(
      params,
      features,
      winner_idx,
      loser_idx,
      dim_continuous,
      dim_categorical,
      num_newton_iters,
  )
  scale = params.comparison_scale
  z = f_mode[winner_idx] - f_mode[loser_idx]
  log_likelihood = jnp.sum(_normal_logcdf(z / scale))
  alpha = jnp.linalg.solve(gram, f_mode)
  # -1/2 f' K^-1 f - 1/2 log|I + W^{1/2} T K T' W^{1/2}|.
  complexity = 0.5 * jnp.dot(f_mode, alpha) + jnp.sum(
      jnp.log(jnp.diag(cholesky))
  )
  laplace_evidence = log_likelihood - complexity
  return -(laplace_evidence + _log_hyperprior(params))


def _prior_state(
    features: PreferenceFeatures,
    winner_idx: jax.Array,
    loser_idx: jax.Array,
    dim_continuous: int,
    dim_categorical: int,
) -> PreferenceGPState:
  """Builds an untrained prior state for when no comparisons are available."""
  num_points = features.num_points
  num_comparisons = winner_idx.shape[0]
  params = PreferenceGPParams.initial(dim_continuous, dim_categorical)
  gram = jnp.zeros((num_points, num_points), dtype=jnp.float64)
  return PreferenceGPState(
      params=params,
      features=features,
      winner_idx=winner_idx,
      loser_idx=loser_idx,
      f_mode=jnp.zeros(num_points, dtype=jnp.float64),
      gram=gram,
      alpha=jnp.zeros(num_points, dtype=jnp.float64),
      cholesky=jnp.zeros(
          (num_comparisons, num_comparisons), dtype=jnp.float64
      ),
      sqrt_w=jnp.zeros(num_comparisons, dtype=jnp.float64),
      w=jnp.zeros(num_comparisons, dtype=jnp.float64),
      dim_continuous=dim_continuous,
      dim_categorical=dim_categorical,
  )


def fit_preference_gp(
    features: PreferenceFeatures,
    winner_idx: jax.Array,
    loser_idx: jax.Array,
    *,
    dim_continuous: int,
    dim_categorical: int,
    rng: jax.Array,
    num_restarts: int = DEFAULT_RESTARTS,
    num_newton_iters: int = DEFAULT_NEWTON_ITERS,
    lbfgs_maxiter: int = DEFAULT_LBFGS_MAXITER,
) -> PreferenceGPState:
  """Fits hyperparameters and the Laplace posterior mode.

  Args:
    features: Unique training points.
    winner_idx: Winner point index per comparison.
    loser_idx: Loser point index per comparison. Same length as `winner_idx`.
    dim_continuous: Number of continuous feature dimensions.
    dim_categorical: Number of categorical feature dimensions.
    rng: PRNG key for multi-restart initialization.
    num_restarts: Number of LBFGS random restarts (the deterministic default
      initialization is always included as one restart).
    num_newton_iters: Fixed Newton iterations inside each Laplace approximation.
    lbfgs_maxiter: Maximum LBFGS iterations per restart.

  Returns:
    A trained `PreferenceGPState`. With zero comparisons a prior state is
    returned without optimization.
  """
  winner_idx = jnp.asarray(winner_idx, dtype=jnp.int32)
  loser_idx = jnp.asarray(loser_idx, dtype=jnp.int32)
  if winner_idx.shape[0] == 0:
    return _prior_state(
        features, winner_idx, loser_idx, dim_continuous, dim_categorical
    )

  objective = functools.partial(
      _neg_log_marginal_likelihood,
      features=features,
      winner_idx=winner_idx,
      loser_idx=loser_idx,
      dim_continuous=dim_continuous,
      dim_categorical=dim_categorical,
      num_newton_iters=num_newton_iters,
  )
  solver = jaxopt.LBFGS(fun=objective, maxiter=lbfgs_maxiter, tol=1e-5)
  jitted_run = jax.jit(solver.run)

  # Always include the deterministic default initialization first.
  init_rngs = jax.random.split(rng, num_restarts)
  initial_params = [
      PreferenceGPParams.initial(dim_continuous, dim_categorical)
  ]
  initial_params.extend(
      PreferenceGPParams.initial(
          dim_continuous, dim_categorical, rng=key, perturbation=0.75
      )
      for key in init_rngs
  )

  best_params = None
  best_objective = jnp.inf
  for init_params in initial_params:
    candidate_params, solver_state = jitted_run(init_params)
    candidate_objective = solver_state.value
    if candidate_objective < best_objective:
      best_objective = candidate_objective
      best_params = candidate_params

  f_mode, cholesky, sqrt_w, w, gram = _laplace_mode(
      best_params,
      features,
      winner_idx,
      loser_idx,
      dim_continuous,
      dim_categorical,
      num_newton_iters,
  )
  alpha = jnp.linalg.solve(gram, f_mode)
  return PreferenceGPState(
      params=best_params,
      features=features,
      winner_idx=winner_idx,
      loser_idx=loser_idx,
      f_mode=f_mode,
      gram=gram,
      alpha=alpha,
      cholesky=cholesky,
      sqrt_w=sqrt_w,
      w=w,
      dim_continuous=dim_continuous,
      dim_categorical=dim_categorical,
  )


def posterior(
    state: PreferenceGPState, test_features: PreferenceFeatures
) -> Tuple[jax.Array, jax.Array]:
  """Returns the Laplace posterior mean and covariance at `test_features`."""
  kernel = build_kernel(
      state.params, state.dim_continuous, state.dim_categorical
  )
  k_ss = _kernel_matrix(
      kernel,
      test_features,
      test_features,
      state.dim_continuous,
      state.dim_categorical,
  )
  if state.num_points == 0:
    # Prior predictive: zero mean, prior covariance.
    cov = k_ss + _JITTER * jnp.eye(
        test_features.num_points, dtype=k_ss.dtype
    )
    return jnp.zeros(test_features.num_points, dtype=k_ss.dtype), cov

  k_s = _kernel_matrix(
      kernel,
      state.features,
      test_features,
      state.dim_continuous,
      state.dim_categorical,
  )
  mean = k_s.T @ state.alpha
  # T K_s, shape (num_comparisons, num_test).
  tk_s = k_s[state.winner_idx] - k_s[state.loser_idx]
  # Woodbury form of (K^{-1} + T' W T)^{-1}:
  # V = K_ss - (T K_s)' W^{1/2} B^{-1} W^{1/2} (T K_s), with
  # ``B = I + W^{1/2} T K T' W^{1/2}`` and ``L L' = B``.
  solved = jax.scipy.linalg.solve_triangular(
      state.cholesky, state.sqrt_w[:, jnp.newaxis] * tk_s, lower=True
  )
  cov = k_ss - solved.T @ solved
  cov = 0.5 * (cov + cov.T) + _JITTER * jnp.eye(
      test_features.num_points, dtype=cov.dtype
  )
  return mean, cov


def utility_mean_std(
    state: PreferenceGPState, test_features: PreferenceFeatures
) -> Tuple[jax.Array, jax.Array]:
  """Returns posterior utility mean and standard deviation."""
  mean, cov = posterior(state, test_features)
  std = jnp.sqrt(jnp.maximum(jnp.diag(cov), 0.0))
  return mean, std


def _pairwise_posterior(
    state: PreferenceGPState,
    features_a: PreferenceFeatures,
    features_b: PreferenceFeatures,
) -> Tuple[jax.Array, jax.Array]:
  """Posterior mean and variance of ``f(a) - f(b)`` for aligned pairs."""
  if features_a.num_points != features_b.num_points:
    raise ValueError(
        'Pairwise prediction requires equally sized feature sets, got '
        f'{features_a.num_points} vs {features_b.num_points}.'
    )
  num_pairs = features_a.num_points
  combined = features_a.concat(features_b)
  mean, cov = posterior(state, combined)
  mean_a = mean[:num_pairs]
  mean_b = mean[num_pairs:]
  var_a = jnp.diag(cov)[:num_pairs]
  var_b = jnp.diag(cov)[num_pairs:]
  cross_ab = cov[:num_pairs, num_pairs:]
  diff_mean = mean_a - mean_b
  diff_var = var_a + var_b - 2.0 * jnp.diag(cross_ab)
  return diff_mean, jnp.maximum(diff_var, 0.0)


def win_probability(
    state: PreferenceGPState,
    features_a: PreferenceFeatures,
    features_b: PreferenceFeatures,
) -> jax.Array:
  """Returns ``P(a beats b)`` under the probit preference model.

  Observation noise (``2 sigma_n^2``) is added to the posterior utility
  difference variance, matching the generative comparison model.
  """
  diff_mean, diff_var = _pairwise_posterior(state, features_a, features_b)
  scale = jnp.sqrt(diff_var + 2.0 * state.params.noise_variance)
  return _normal_cdf(diff_mean / scale)


def utility_difference(
    state: PreferenceGPState,
    features_a: PreferenceFeatures,
    features_b: PreferenceFeatures,
) -> Tuple[jax.Array, jax.Array]:
  """Returns posterior mean and standard deviation of ``f(a) - f(b)``."""
  diff_mean, diff_var = _pairwise_posterior(state, features_a, features_b)
  return diff_mean, jnp.sqrt(diff_var)


def thompson_samples(
    state: PreferenceGPState,
    test_features: PreferenceFeatures,
    num_samples: int,
    rng: jax.Array,
) -> jax.Array:
  """Draws posterior utility Thompson samples.

  Args:
    state: Trained preference GP state.
    test_features: Candidate points, shape ``(num_candidates, ...)``.
    num_samples: Number of independent draws.
    rng: PRNG key.

  Returns:
    Array of shape ``(num_candidates, num_samples)``.
  """
  mean, cov = posterior(state, test_features)
  chol = _retrying_cholesky(cov)
  standard_draws = jax.random.normal(
      rng, shape=(test_features.num_points, num_samples), dtype=cov.dtype
  )
  return mean[:, jnp.newaxis] + chol @ standard_draws
