import numpy as np


def svd(matrix):
    m, n = matrix.shape

    matrix_product_left = np.dot(matrix, matrix.T)
    eigenvalues_left, eigenvectors_left = np.linalg.eigh(matrix_product_left)

    idx = np.argsort(eigenvalues_left)[::-1]
    eigenvalues_left = eigenvalues_left[idx]
    u = eigenvectors_left[:, idx]

    singular_values = np.sqrt(eigenvalues_left)
    k = min(m, n)
    sigma = np.zeros((m, n))
    np.fill_diagonal(sigma, singular_values[:k])

    matrix_product_right = np.dot(matrix.T, matrix)
    eigenvalues_right, eigenvectors_right = np.linalg.eigh(matrix_product_right)

    idx = np.argsort(eigenvalues_right)[::-1]
    v = eigenvectors_right[:, idx]

    for i in range(k):
        if singular_values[i] > 0:
            u[:, i] = np.dot(matrix, v[:, i]) / singular_values[i]

    reconstructed_matrix = np.dot(u, np.dot(sigma, v.T))
    check = np.allclose(matrix, reconstructed_matrix)
    if check:
        print("SVD is correct")
    else:
        print("SVD is incorrect")

    return u, sigma, v


matrix = np.array([[1, 2], [2, 1], [3, 4]])
u, sigma, v = svd(matrix)
print("U:\n", u)
print("Sigma:\n", sigma)
print("V:\n", v)
