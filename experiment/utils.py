# this utils file is composed of select modified functions from benchmarking/utils.py or the google-deepmind/alphatensor repo
# the functions are modified for use with the experiment 

from typing import Callable, List, Tuple
import numpy as np

BlockMatrix = List[List[np.ndarray]]

def _get_n_from_factors(factors: np.ndarray) -> int:
  """Computes the matrix multiplication tensor size n based on `factors`.

  E.g. when multiplying 2x2 matrices with Strassen, the `factors` are of shape
  [4, 7], and this function will return 2.

  Args:
    factors: [3, n^2, R] shaped NumPy array representing a factorization of T_n.
  Returns:
    n, the size of matrices being multiplied by the algorithm represented by
    `factors`.
  """
  u, v, w = factors
  # Assert that the tensor is a cube.
  assert u.shape[0] == v.shape[0]
  assert u.shape[0] == w.shape[0]
  n = int(np.sqrt(u.shape[0]))
  assert u.shape[0] == n ** 2
  return n


def algorithm_from_factors(factors: np.ndarray) -> Callable[[BlockMatrix, BlockMatrix], BlockMatrix]:
  """Returns a function implementing the algorithm described by `factors`.

  Args:
    factors: Matricized factorization of a matrix multiplication tensor, i.e.
      an array of shape [3, n, n, rank].
  Returns:
    Function, which given two block matrices `a` and `b` returns the block
    matrix `c` given by `c = a @ b`.
  """
  assert factors[0].shape[0] == factors[1].shape[0]
  assert factors[1].shape[0] == factors[2].shape[0]
  factors = [factors[0].copy(), factors[1].copy(), factors[2].copy()]
  n = int(np.sqrt(factors[0].shape[0]))
  rank = factors[0].shape[-1]
  factors[0] = factors[0].reshape(n, n, rank)
  factors[1] = factors[1].reshape(n, n, rank)
  factors[2] = factors[2].reshape(n, n, rank)
  # The factors are for the transposed (symmetrized) matrix multiplication
  # tensor. So to use the factors, we need to transpose back.
  factors[2] = factors[2].transpose(1, 0, 2)

  def f(a: BlockMatrix, b: BlockMatrix) -> BlockMatrix:
    """Multiplies block matrices `a` and `b`."""
    n = len(a)
    result = [[None] * n for _ in range(n)]
    for alpha in range(rank):
      left = None
      for i in range(n):
        for j in range(n):
          if factors[0][i, j, alpha] != 0:
            curr = factors[0][i, j, alpha] * a[i][j]
            if left is None:
              left = curr
            else:
              left += curr
      right = None
      for j in range(n):
        for k in range(n):
          if factors[1][j, k, alpha] != 0:
            curr = factors[1][j, k, alpha] * b[j][k]
            if right is None:
              right = curr
            else:
              right += curr

      matrix_product = left @ right

      for i in range(n):
        for k in range(n):
          if factors[2][i, k, alpha] != 0:
            curr = factors[2][i, k, alpha] * matrix_product
            if result[i][k] is None:
              result[i][k] = curr
            else:
              result[i][k] += curr
    return result

  return f

def _generate_random_matrices(matrix_dims: Tuple[int, int, int],seed: int) -> Tuple[np.ndarray, np.ndarray]:
  """Generates two random NumPy matrices to be multiplied."""
  np.random.seed(seed)
  a = np.random.randn(matrix_dims[0], matrix_dims[1])
  b = np.random.randn(matrix_dims[1], matrix_dims[2])
  return a, b

def _generate_random_int_matrices(matrix_dims: Tuple[int, int, int],seed: int) -> Tuple[np.ndarray, np.ndarray]:
  """Generates two random NumPy matrices to be multiplied."""
  np.random.seed(seed)
  a = np.random.randint(1,256,(matrix_dims[0], matrix_dims[1]))
  b = np.random.randint(1,256,(matrix_dims[1], matrix_dims[2]))
  return a, b

def block_split(matrix: np.ndarray, n_rows: int, n_cols: int) -> BlockMatrix:
  """Splits `matrix` into a `n_rows x n_cols` block matrix."""
  rows = np.split(matrix, n_rows, axis=0)
  return [np.split(row, n_cols, axis=1) for row in rows]

# takes n,m,p dimensions of two matrices multiplied together
# returns numpy.ndarray
def generate_naive_factorization(matrix_dims: Tuple[int, int, int]):
    n,m,p = matrix_dims
    multiplications = n * m *p
    u = np.zeros(((n*m),multiplications ),dtype=np.int32)
    v = np.zeros(((m*p),multiplications ),dtype=np.int32)
    w = np.zeros(((n*p),multiplications ),dtype=np.int32)
    
    u_flat = u.flatten()

    
    multiplication_count =0
    c_index = 0
    #loop over matrix c row
    for i in range(0,n):
        for z in range(0,p):
            for x in range(0,int(multiplications/(n*p))):
                u[(z*m)+(x)][multiplication_count] = 1
                v[i+(x*p)][multiplication_count]=1
                w[c_index][multiplication_count] = 1

                multiplication_count = multiplication_count + 1
            c_index = c_index + 1
    return np.stack((u,v,w),axis=0)

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

def pad(array, reference_shape, offsets):
    """
    array: Array to be padded
    reference_shape: tuple of size of ndarray to create
    offsets: list of offsets (number of elements must be equal to the dimension of the array)
    will throw a ValueError if offsets is too big and the reference_shape cannot handle the offsets
    """

    # Create an array of zeros with the reference shape
    result = np.zeros(reference_shape,dtype=np.int32)
    # Create a list of slices from offset to offset + shape in each dimension
    insertHere = [slice(offsets[dim], offsets[dim] + array.shape[dim]) for dim in range(array.ndim)]
    # Insert the array in the result at the specified offsets
    result[tuple(insertHere)] = array
    return result