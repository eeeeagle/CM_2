import numpy as np


def print_float_matrix(m, a, b):
    for i in range(a):
        print("[", end=' ')
        for j in range(b):
            print("%8.3f" % m[i][j], end=' ')
        print("]\n", end=' ')


def print_int_matrix(m, a, b):
    for i in range(a):
        print("[", end=' ')
        for j in range(b):
            print("%4d" % m[i][j], end=' ')
        print("]\n", end=' ')


""" 
TASK 1
"""
print("===================================\nTASK 1\n")

M = np.random.uniform(-1, 1, (8, 8))
print("Random float matrix:\n", end=' ')
print_float_matrix(M, 8, 8)

d = 0
for i in range(8):
    d += (M[:, 0])[i] * (M[7, :])[i]
print("\nDot product = %.5f" % d)

""" 
TASK 2
"""
print("\n===================================\nTASK 2\n")

n = np.random.randint(3, 8)
k = np.random.randint(3, 8)
m = np.random.randint(3, 8)

A = np.random.randint(-6, 7, (n, k))
print("Matrix A:\n", end=' ')
print_int_matrix(A, n, k)

B = np.random.randint(-6, 7, (k, m))
print("\nMatrix B:\n", end=' ')
print_int_matrix(B, k, m)

C = np.zeros((n, m), int)
for i in range(n):
    for j in range(m):
        for p in range(k):
            C[i, j] = C[i, j] + A[i, p]*B[p, j]
print("\n1) Scalar algorithm A*B:\n", end=' ')
print_int_matrix(C, n, m)

for i in range(n):
    for j in range(m):
            C[i, j] = np.dot(A[i, :], B[:, j])
print("\n2) Vector algorithm A*B:\n", end=' ')
print_int_matrix(C, n, m)

C = np.dot(A, B)
print("\n3) np.dot(A, B):\n", end=' ')
print_int_matrix(C, n, m)

""" 
TASK 3
"""
print("\n===================================\nTASK 3\n")

A = np.random.randint(-11, 11, (7, 7))
A = np.tril(A)
A[np.diag_indices(7)] = 1
print("Matrix A:\n", end=' ')
print_int_matrix(A, 7, 7)

B = np.random.randint(-6, 6, (7, 1))
print("\nVector B =\n", end=' ')
print_int_matrix(B, 7, 1)

X = np.zeros((7, 1), float)
X[0] = B[0]
for i in range(1, 7):
    X[i] = B[i] - np.dot(A[i, :i], X[:i])

print("\nX =\n", end=' ')
print_float_matrix(X, 7, 1)

X_CHECK = np.linalg.solve(A, B)
X_CHECK = np.array(X_CHECK.tolist(), float)
print("\nnp.linalg.solve(A, B) = \n", end=' ')
print_float_matrix(X_CHECK, 7, 1)

""" 
TASK 4
"""
print("\n===================================\nTASK 4\n")

A = np.array([[1.7, -1.8, 1.9, -57.4],
              [1.1, -4.3, 1.5, -1.7],
              [1.2, 1.4, 1.6, 1.8],
              [7.1, -1.3, -4.1, 5.2]], float)
print("Matrix A:\n", end=' ')
print_float_matrix(A, 4, 4)

B = np.array([[10], [19], [20], [10]], float)
print("\nVector B = \n", end=' ')
print_int_matrix(B, 4, 1)

U = np.zeros((4, 4), float)
L = np.identity(4, float)

for i in range(4):
    for j in range(4):
        if i <= j:
            U[i, j] = A[i, j] - np.dot(L[i, :i], U[:i, j])
        if i > j:
            L[i, j] = (A[i, j] - np.dot(L[i, :j], U[:j, j])) / U[j, j]

print("\nL =\n", end=' ')
print_float_matrix(L, 4, 4)

print("\nU =\n", end=' ')
print_float_matrix(U, 4, 4)

print("\nLU =\n", end=' ')
print_float_matrix(np.dot(L, U), 4, 4)

Y = B.copy()
Y[0] = Y[0] / L[0, 0]
for i in range(1, len(Y)):
    Y[i] = (Y[i] - np.dot(L[i, :i], Y[:i])) / L[i, i]
print("\nLY = B\nY =\n", end=' ')
print_float_matrix(Y, 4, 1)

X = Y.copy()
X[-1] = X[-1] / U[-1, -1]
for i in range(len(X) - 2, -1, -1):
    X[i] = (X[i] - np.dot(U[i, i + 1:], X[i + 1:])) / U[i, i]
print("\nUX = y\nX =\n", end=' ')
print_float_matrix(X, 4, 1)

X_CHECK = np.linalg.solve(A, B)
X_CHECK = np.array(X_CHECK.tolist(), float)
print("\nnp.linalg.solve(A, B) = \n", end=' ')
print_float_matrix(X_CHECK, 4, 1)
