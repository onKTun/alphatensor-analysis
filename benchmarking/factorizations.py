# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Factorizations discovered by AlphaTensor and the Strassen^2 factorization."""

import numpy as np


def get_4x4x4_alphatensor_gpu() -> np.ndarray:
  """Returns a factorization for fast matrix multiplication on NVIDIA V100 GPUs.

  This factorization was discovered by AlphaTensor while optimizing for the
  runtime of multiplying two 8192 x 8192 matrices on an NVIDIA V100 GPU in
  `float32`.

  Returns:
    [3, 16, 49]-shaped array representing a rank-49 factorization of the
    (symmetrized version of the) matrix multiplication tensor T_4 = <4, 4, 4>
    in standard arithmetic.
  """
  u = np.array([
      [-1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
       1, 0, -1, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0,
       0, 0, 0],
      [1, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 1, 0, 0, 1, -1, 0, 0, 0, 0, 0,
       0, 0, 0, 1, 0, -1, 0, -1, 0, 1, 0, 0, 0, 0, -1, 1, 1, 0, 1, 0, 1, 0, 0,
       1, 0, 0],
      [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1,
       0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -1,
       0],
      [-1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, -1, 0, 1, 0, 1, 0, -1, 1, 0, 0, 0, 1, 0, -1, 0, -1, 0, -1, 0,
       0, -1, 1, 0],
      [-1, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, -1, 0, -1, 0, 1, 0, 1, 0, 1, 0, 0, 0,
       0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, -1, 0, -1, 1, 0, 0,
       -1, 0, 0, 0],
      [1, 0, 0, 1, 0, 0, 0, -1, 1, 0, 0, 0, 0, 1, 0, 0, 0, -1, 0, -1, 0, 0, 0,
       0, 0, 0, 1, 0, 0, 0, -1, 0, 1, 0, 1, 0, 0, -1, 1, 1, 0, 1, 0, 0, 0, 1,
       0, 0, 0],
      [1, 1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, -1, 0, 0, 0,
       0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 1, 0, 1, -1, 0, 0, 1,
       0, -1, 0],
      [-1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 1, 0, 0, 0,
       0, 0, 0, -1, 0, 0, 0, 1, 0, -1, 1, -1, 1, 0, 1, 0, -1, 0, -1, 0, 0, 0,
       -1, 0, 1, 0],
      [-1, 0, 0, 0, -1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
       1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0,
       0, 0, 0],
      [1, 0, 1, 0, 0, 0, 0, -1, 1, 0, -1, 0, -1, 1, 0, 0, 1, -1, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, -1, 1, 0, 0, 1, 0, 1, 0, 0,
       1, 0, 0],
      [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
       0],
      [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 1, -1,
       0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       -1, 0, 0],
      [-1, 0, 0, -1, 0, 0, 0, 1, 0, 0, 1, -1, 1, -1, 0, 1, 0, 1, 0, 1, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1, 0, -1, 0, 0, -1, 1, 0, 0, 0,
       0, 0, 1],
      [1, 0, 0, 1, 0, 0, 0, -1, 1, 0, -1, 0, -1, 1, 0, 0, 0, -1, 0, -1, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 1, 0, -1, -1, 1, 0, 0, 1, 0, 0, 0,
       0, 0, 0, 0],
      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
       1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0,
       0],
      [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, -1,
       0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
       0, 0, 0]], dtype=np.int32)
  v = np.array([
      [0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
       -1, 0, -1, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0,
       0, 0, 0],
      [0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, -1, 0, 1, 0, 0, 0, 0, 0, 0,
       0, 0, -1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, -1, -1, -1, 0, 1, 1, 0, 0, 0, 0,
       1, 0],
      [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
       0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
       -1],
      [0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 1, -1, 0, 0, -1, 1, 0, -1, 0, 0, 0, 1, 0, 1, 0, -1, -1, 0, 0,
       0, 0, -1, 1],
      [0, -1, 0, 1, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, -1, 1, 1, 0, 0, 1, 0, 0, 0,
       0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 1, 0, 0, -1, 1, 0, -1,
       0, 0, 0],
      [0, 1, 0, -1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, -1, 0, 0, 0,
       0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, -1, -1, -1, 0, 0, 1, 0, 0, 1,
       0, 0, 0],
      [0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0,
       0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, -1, 0, 0, 1, -1, 0, 1, 0,
       0, -1],
      [0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0,
       0, 0, 0, 0, -1, 0, 0, -1, 1, 0, -1, 0, -1, 1, 1, 0, 1, 0, 0, -1, 0, 0,
       -1, 0, 0, 1],
      [0, -1, 0, 0, 1, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
       -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0,
       0, 0, 0],
      [1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, -1, 0, -1, 1, -1, 0, 1, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, -1, -1, 0, 0, 1, 1, 0, 0, 0,
       0, 1, 0],
      [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
       0],
      [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, -1, 0, 1,
       0, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, -1, 0],
      [0, -1, 0, 1, 0, 0, 0, -1, 0, -1, 0, 1, 0, 1, -1, 1, 1, 0, 0, 1, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 1, 0, 0, 0, -1, 1, 0, 0,
       1, 0, 0],
      [0, 1, 0, -1, 0, 0, 0, 1, 1, 0, 0, -1, 0, -1, 1, -1, 0, 0, 0, -1, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, -1, 1, 0, -1, -1, 0, 0, 0, 1, 0, 0,
       0, 0, 0, 0],
      [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0,
       1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
       0],
      [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, -1, 0, 0,
       0, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0,
       0, 0, 0]], dtype=np.int32)
  w = np.array([
      [0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
       1, 0, 1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0,
       0, 0, 0],
      [0, 0, 1, 0, 0, 0, 0, -1, 1, 0, 0, 0, 1, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, -1, 1, 0, -1, 1, 0, 0, 0, 0, 0, -1, 1, 1, 0, 0, 1, 1, 0, 0,
       0, 0, 1],
      [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1,
       0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1,
       0, 0],
      [0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 1, -1, 0, 1, -1, 1, 0, 0, 0, 0, 1, 0, -1, 0, 0, -1, -1, 0,
       0, 1, 0, -1],
      [0, 0, -1, 1, 0, 0, 0, 1, 0, 0, -1, 0, -1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0,
       0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1, -1, 0, 1, 0, -1, 0,
       -1, 0, 0, 0],
      [0, 0, 1, -1, 0, 0, 0, -1, 1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 0, -1, 1, 0, 0, 0, 0, 1, -1, 1, 1, 0, 0, 0, 1, 0, 1,
       0, 0, 0],
      [1, 0, 1, 0, 0, 0, 0, -1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, -1, 0, 0, 0,
       0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 1, 0, -1, 0, 1, 0, 1,
       -1, 0, 0],
      [0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
       0, 0, 0, 0, 0, -1, 0, 1, -1, 1, 0, 1, 0, -1, 1, 0, -1, 0, 0, 0, -1, 0,
       -1, 1, 0, 0],
      [0, 0, -1, 0, 0, 1, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
       1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0,
       0, 0, 0],
      [0, 1, 1, 0, 0, 0, 0, -1, 1, -1, 0, 0, 1, 0, -1, 1, -1, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, -1, 1, 0, 0, 0, 1, 1, 0, 0,
       0, 0, 1],
      [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
       0],
      [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 1, -1, 0,
       0, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, -1],
      [0, 0, -1, 1, 0, 0, 0, 1, 0, 1, -1, 0, -1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, -1, 0, 0, 1, 0, -1, 0, 0,
       0, 1, 0],
      [0, 0, 1, -1, 0, 0, 0, -1, 1, -1, 0, 0, 1, 0, -1, 0, -1, 0, 0, -1, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1, 0, -1, 1, -1, 1, 0, 0, 0, 0, 1, 0,
       0, 0, 0, 0],
      [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
       1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
       0],
      [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0,
       0, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0,
       0, 0, 0]], dtype=np.int32)
  return np.array([u, v, w])


def get_4x4x4_alphatensor_tpu() -> np.ndarray:
  """Returns a factorization for fast matrix multiplication on a Google TPUv2.

  This factorization was discovered by AlphaTensor while optimizing for the
  runtime of multiplying two 8192 x 8192 matrices on a TPUv2 in `bfloat16`.

  Returns:
    [3, 16, 49]-shaped array representing a rank-49 factorization of the
    (symmetrized version of the) matrix multiplication tensor T_4 = <4, 4, 4>
    in standard arithmetic.
  """
  u = np.array([
      [1, 1, 0, 0, 0, 1, -1, 1, -1, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
       0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, -1, 0, 1, 0,
       0, 0],
      [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, -1, 0, 1, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
       0, 0],
      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0,
       1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 1, 0,
       -1, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 1, 0,
       0, 0],
      [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 0,
       1],
      [1, 0, 1, -1, 1, 0, 0, 1, -1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1,
       0, 0, 0, 1, 0, -1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 1],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0],
      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
       0],
      [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
       0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0,
       0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1,
       0, 0],
      [1, 1, 0, 0, 0, 1, -1, 0, 0, 1, 0, 1, -1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0,
       1, 0],
      [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, -1,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1, 0,
       0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0,
       0],
      [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, -1,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
       0, 0],
      [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, 0, 0,
       0, 1, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
       0, 0],
      [1, 0, 1, -1, 1, 0, 0, 0, 0, 1, 0, 1, -1, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0,
       1, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, -1, 0,
       1, 0, 0]], dtype=np.int32)
  v = np.array([
      [1, 0, 1, 0, -1, 0, 1, 0, 1, -1, 0, 1, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 1,
       0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, -1, 0, 0,
       0, 0],
      [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, -1, 0, 1, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
       0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 1, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 0,
       -1],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0,
       0, 0],
      [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
       0],
      [1, -1, 0, 1, 0, 1, 0, 0, 1, -1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0,
       1, 0, 0, 0, 1, 0, -1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
       0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0,
       0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0],
      [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
       0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
       0, -1, 0, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0,
       0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       -1, 0],
      [1, 0, 1, 0, -1, 0, 1, 1, 0, 0, -1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0,
       0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0,
       0, 1],
      [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0,
       -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
       -1, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0,
       0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0,
       -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
       0, 1, 0],
      [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
       0, 0],
      [1, -1, 0, 1, 0, 1, 0, 1, 0, 0, -1, 0, 1, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
       0, 1, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, -1,
       0, 1, 0]], dtype=np.int32)
  w = np.array([
      [1, 0, 0, 1, 1, -1, 0, -1, 0, 1, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0,
       1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, -1, 0,
       0, 0],
      [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, -1, 0, 0, 0, 0, 1, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0,
       0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 1,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 1, 0, -1,
       0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
       0, 0],
      [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0,
       1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 1, 0, 1,
       0],
      [1, 1, -1, 0, 0, 0, 1, -1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
       0, 1, 1, 0, 0, 0, 0, -1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       1, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
       0],
      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
       0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0,
       0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       -1],
      [1, 0, 0, 1, 1, -1, 0, 0, 1, 0, 1, -1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1,
       0, 0],
      [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, -1, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       -1],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0,
       0, 0],
      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0,
       -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
       0, 1],
      [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
       0, 0],
      [1, 1, -1, 0, 0, 0, 1, 0, 1, 0, 1, -1, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 1,
       0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1, 0, -1, 0, 0,
       0, 0, 1]], dtype=np.int32)
  return np.array([u, v, w])


def _get_2x2x2_strassen() -> np.ndarray:
  """Returns [3, 4, 7] array, representing a rank-7 factorization of T_2."""

  # List of 7 factors, each of shape [3, 4].
  factors = [[[1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]],
             [[1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]],
             [[0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]],
             [[0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]],
             [[0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]],
             [[-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]],
             [[1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]]]

  # Transpose into our standard format [3, S, R] = [3, 4, 7],
  return np.transpose(np.array(factors, dtype=np.int32), [1, 2, 0])


def _product_factors(factors1: np.ndarray, factors2: np.ndarray) -> np.ndarray:
  """Computes the Kronecker product of `factors1` and `factors2`.

  Args:
    factors1: [3, n1**2, R1] factors of a tensor T1
    factors2: [3, n2**2, R2] factors of a tensor T2

  Returns:
    [3, n1**2 * n2 ** 2, R1 * R2] factorization of the Kronecker square tensor
    Reshape(kron(RT1, RT2)), where `RT1` and `RT2` are the reshapes of T1 and T2
    into 6-dimensional tensors, and `Reshape` reshapes the tensor back into a
    3-dimensional one.
  """
  _, side1, rank1 = np.shape(factors1)
  _, side2, rank2 = np.shape(factors2)

  n1 = int(np.round(np.sqrt(side1)))
  n2 = int(np.round(np.sqrt(side2)))

  if n1 * n1 != side1 or n2 * n2 != side2:
    raise ValueError(f'The sides {side1}, {side2} of factors passed to '
                     '`product_factors` must be both perfect squares.')
  product = np.einsum('...abi,...cdj->...acbdij',
                      factors1.reshape((3, n1, n1, rank1)),
                      factors2.reshape((3, n2, n2, rank2))
                     )  # [3, n1, n2, n1, n2, R1, R2]
  return np.reshape(product, (3, n1 * n2 * n1 * n2, rank1 * rank2))


def get_4x4x4_strassen_squared() -> np.ndarray:
  """Returns Strassen^2 factorization for fast multiplication of 4x4 matrices.

  This factorization is obtained by squaring (recursively applying twice)
  Strassen's rank-7 factorization of T_2.

  Returns:
    [3, 16, 49]-shaped array representing a rank-49 factorization of the
    (symmetrized version of the) matrix multiplication tensor T_4 = <4, 4, 4>
    in standard arithmetic.
  """
  strassen = _get_2x2x2_strassen()  # [3, 4, 7]
  return _product_factors(strassen, strassen)  # [3, 16, 49]