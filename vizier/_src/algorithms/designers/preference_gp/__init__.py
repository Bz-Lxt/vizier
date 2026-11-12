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

"""Preference learning with a Gaussian Process over pairwise win/loss data.

This subpackage implements a probit-preference GP whose latent utility function
and kernel hyperparameters are learned from pairwise comparisons, without
requiring numeric objectives. See `preference_gp.py` for the JAX math core and
`preference_gp_designer.py` for the `vza.Designer` wrapper.
"""
