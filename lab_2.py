import numpy as np


def eigenvalues_and_eigenvectors(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    for i in range(len(eigenvalues)):
        Av = np.dot(matrix, eigenvectors[:, i])
        lv = eigenvalues[i] * eigenvectors[:, i]

        if np.allclose(Av, lv):
            print(f"Eigenvalue {eigenvalues[i]} and eigenvector {eigenvectors[:, i]} are correct")
        else:
            print(f"Eigenvalue {eigenvalues[i]} and eigenvector {eigenvectors[:, i]} are incorrect")
    return eigenvalues, eigenvectors


my_matrix = np.array([[1, 2], [2, 1]])
eigenvalues_and_eigenvectors(my_matrix)

