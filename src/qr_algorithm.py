import numpy as np
from scipy.linalg import hessenberg
import sys


def run(matrix, tol=1e-10, max_iters=1000):
    """
    Perform the QR Algorithm to find the eigenvalues of a given matrix.

    Parameters
    ----------
    matrix : np.ndarray
        The matrix to find the eigenvalues of.
    tol : float
        The tolerance for the algorithm to stop.
    max_iters : int
        The maximum number of iterations to perform.

    Returns
    -------
    np.ndarray
        The eigenvalues of the given matrix.
    """
    assert matrix.shape[0] == matrix.shape[1], "Matrix must be square"

    cur_matrix = hessenberg(matrix)
    for _ in range(max_iters):
        Q, R = np.linalg.qr(cur_matrix)
        next_matrix = R @ Q

        # Check Convergance
        if np.allclose(cur_matrix, next_matrix, atol=tol):
            return np.diag(cur_matrix)

        cur_matrix = next_matrix

    print("Warning: Maximum iterations reached without converging.")
    return np.diag(cur_matrix)


if __name__ == "__main__":
    A1 = np.loadtxt('../input/A1.txt')
    A2 = np.loadtxt('../input/A2.txt')
    A3 = np.load('../input/A3.npy')

    print("Eigenvalues of A1: ", run(A1))
    print("Eigenvalues of A2: ", run(A2))
    print("Eigenvalues of A3: ", run(A3))
