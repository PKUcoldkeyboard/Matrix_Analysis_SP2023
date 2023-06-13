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
        if np.all(np.abs(next_matrix[np.triu_indices_from(next_matrix, k=-1)]) < tol):
            cur_matrix = next_matrix
            break

        cur_matrix = next_matrix

    print("Warning: Maximum iterations reached without converging.")
    return cur_matrix


if __name__ == "__main__":
    A1 = np.loadtxt('./input/A1.txt')
    A2 = np.loadtxt('./input/A2.txt')
    A3 = np.load('./input/A3.npy')

    ret_A1 = run(A1)
    ret_A2 = run(A2)
    ret_A3 = run(A3)

    np.set_printoptions(precision=10, suppress=True)
    print("Result of A1: \n", ret_A1)
    print("Result of A2: \n", ret_A2)
    print("Result of A3: \n", ret_A3)
    print("Result of A1: \n", np.diag(ret_A1))
    print("Result of A2: \n", np.diag(ret_A2))
    print("Result of A3: \n", np.diag(ret_A3))
