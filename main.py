import numpy as np
import matplotlib.pyplot as plt


def plot_objects(points1, points2, title1, title2):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(points1[:, 0], points1[:, 1], marker='o')
    plt.title(title1)
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(points2[:, 0], points2[:, 1], marker='o')
    plt.title(title2)
    plt.grid(True)
    plt.show()


def object_rotation(points, angle):
    angle = np.radians(angle)
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return np.dot(rotation_matrix, points.T).T


triangle = np.array([[0, 0], [1, 0], [0.2, 1], [0, 0]])
trapezoid = np.array([[0, 0], [0, 1], [2, 1], [3, 0], [0, 0]])

trapezoid_rotated = object_rotation(trapezoid, 90)
plot_objects(trapezoid, trapezoid_rotated, 'Trapezoid', 'Rotated Trapezoid')

triangle_rotated = object_rotation(triangle, 90)
plot_objects(triangle, triangle_rotated, 'Triangle','Rotated Triangle')
