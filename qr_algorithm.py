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
    tol : float, optional
        The tolerance for the algorithm to stop (default is 1e-10).
    max_iters : int, optional
        The maximum number of iterations to perform (default is 1000).

    Returns
    -------
    np.ndarray
        The eigenvalues of the given matrix.
    """
    # Check if the matrix is square
    assert matrix.shape[0] == matrix.shape[1], "Matrix must be square"

    # Transform the matrix to Hessenberg form
    cur_matrix = hessenberg(matrix)

    # Perform QR algorithm until convergence or max iteration is reached
    for _ in range(max_iters):
        Q, R = np.linalg.qr(cur_matrix)
        next_matrix = R @ Q

        # Check convergence by comparing all elements below the diagonal with the tolerance
        if np.all(np.abs(next_matrix[np.triu_indices_from(next_matrix, k=-1)]) < tol):
            cur_matrix = next_matrix
            break

        cur_matrix = next_matrix

    print(f"ConvergenceWarning: Maximum number of iterations {max_iters} reached. Increase it to improve convergence.")
    return cur_matrix


if __name__ == "__main__":
    # Load matrices from files
    A1 = np.loadtxt('./input/A1.txt')
    A2 = np.loadtxt('./input/A2.txt')
    A3 = np.load('./input/A3.npy')

    # Apply the QR algorithm to each matrix
    ret_A1 = run(A1)
    ret_A2 = run(A2)
    ret_A3 = run(A3)

    # Configure print options for numpy
    np.set_printoptions(precision=10, suppress=True)

    # Print results
    print("Result of A1: \n", ret_A1)
    print("Result of A2: \n", ret_A2)
    print("Result of A3: \n", ret_A3)
    print("Result of A1: \n", np.diag(ret_A1))
    print("Result of A2: \n", np.diag(ret_A2))
    print("Result of A3: \n", np.diag(ret_A3))
