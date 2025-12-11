import numpy as np

def lu_factorisation(A):
    """
    Decompose A into L and U such that A = L @ U.
    L is lower triangular with 1s on the diagonal.
    U is upper triangular.
    """
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square")

    L = np.eye(n, dtype=float)   # start with identity for L
    U = np.zeros((n, n), dtype=float)

    for row in range(n):
        # compute U[row, col] for col >= row
        for col in range(row, n):
            sum_val = 0.0
            for k in range(row):
                sum_val += L[row, k] * U[k, col]
            U[row, col] = A[row, col] - sum_val

        # compute L[col, row] for col > row
        for col in range(row + 1, n):
            sum_val = 0.0
            for k in range(row):
                sum_val += L[col, k] * U[k, row]
            L[col, row] = (A[col, row] - sum_val) / U[row, row]

    return L, U