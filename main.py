import numpy as np
import matplotlib.pyplot as plt


def plot_objects(points, title):
    plt.plot(points[:, 0], points[:, 1])
    plt.title(title)
    plt.grid(True)
    plt.show()


triangle = np.array([[0, 0], [1, 0], [0.2, 1], [0, 0]])
trapezoid = np.array([[0, 0], [2, 0], [3, 1], [0, 1], [0, 0]])
plot_objects(triangle, 'Triangle')
plot_objects(trapezoid, 'Trapezoid')
