import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt


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

file_path = 'ratings.csv'
df = pd.read_csv(file_path)

ratings_matrix = df.pivot(index='userId', columns='movieId', values='rating')

ratings_matrix = ratings_matrix.dropna(thresh=200, axis=0)
ratings_matrix = ratings_matrix.dropna(thresh=90, axis=1)

ratings_matrix_filled = ratings_matrix.fillna(2.5)
R = ratings_matrix_filled.values
user_ratings_mean = np.mean(R, axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

U, sigma, Vt = svds(R_demeaned, k=3)

print("U:\n", U)
print("Sigma:\n", sigma)
print("Vt:\n", Vt)

num_users_to_plot = 20

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(U[:num_users_to_plot, 0], U[:num_users_to_plot, 1], U[:num_users_to_plot, 2])

ax.set_title('Users')
plt.show()

V = Vt.T
num_movies_to_plot = 20

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(num_movies_to_plot):
    ax.scatter(V[i, 0], V[i, 1], V[i, 2], label=f'Movie {i+1}')

ax.set_title('Films')
plt.show()
