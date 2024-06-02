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


def object_scaling(points, scale):
    scaling_matrix = np.array([[scale, 0], [0, scale]])
    return np.dot(points, scaling_matrix)


def object_axis_reflection(points, axis):
    if axis == 'x':
        reflection_matrix = np.array([[1, 0], [0, -1]])
    else:
        reflection_matrix = np.array([[-1, 0], [0, 1]])
    return np.dot(points, reflection_matrix)


def object_shear(points, shear_factor, axis):
    if axis == 'x':
        shear_matrix = np.array([[1, shear_factor], [0, 1]])
    else:
        shear_matrix = np.array([[1, 0], [shear_factor, 1]])
    return np.dot(shear_matrix, points.T).T


triangle = np.array([[0, 0], [1, 0], [0.2, 1], [0, 0]])
trapezoid = np.array([[0, 0], [0, 1], [2, 1], [3, 0], [0, 0]])

trapezoid_rotated = object_rotation(trapezoid, 90)
plot_objects(trapezoid, trapezoid_rotated, 'Trapezoid', 'Rotated Trapezoid')

triangle_rotated = object_rotation(triangle, 90)
plot_objects(triangle, triangle_rotated, 'Triangle','Rotated Triangle')

scale_factor = 2

trapezoid_scaled = object_scaling(trapezoid, scale_factor)
plot_objects(trapezoid, trapezoid_scaled, 'Trapezoid', f'Scaled Trapezoid (scale factor={scale_factor})')

triangle_scaled = object_scaling(triangle, scale_factor)
plot_objects(triangle, triangle_scaled, 'Triangle', f'Scaled Triangle (scale factor={scale_factor})')

trapezoid_reflected_x = object_axis_reflection(trapezoid, 'x')
plot_objects(trapezoid, trapezoid_reflected_x, 'Trapezoid', 'Reflected Trapezoid (X-axis)')

triangle_reflected_x = object_axis_reflection(triangle, 'x')
plot_objects(triangle, triangle_reflected_x, 'Triangle', 'Reflected Triangle (X-axis)')

trapezoid_reflected_y = object_axis_reflection(trapezoid, 'y')
plot_objects(trapezoid, trapezoid_reflected_y, 'Trapezoid', 'Reflected Trapezoid (Y-axis)')

triangle_reflected_y = object_axis_reflection(triangle, 'y')
plot_objects(triangle, triangle_reflected_y, 'Triangle', 'Reflected Triangle (Y-axis)')


shear_factor = 2
trapezoid_sheared_x = object_shear(trapezoid, shear_factor, 'x')
plot_objects(trapezoid, trapezoid_sheared_x, 'Trapezoid', f'Sheared Trapezoid (X-axis, shear factor={shear_factor})')

triangle_sheared_x = object_shear(triangle, shear_factor, 'x')
plot_objects(triangle, triangle_sheared_x, 'Triangle', f'Sheared Triangle (X-axis, shear factor={shear_factor})')

trapezoid_sheared_y = object_shear(trapezoid, shear_factor, 'y')
plot_objects(trapezoid, trapezoid_sheared_y, 'Trapezoid', f'Sheared Trapezoid (Y-axis, shear factor={shear_factor})')

triangle_sheared_y = object_shear(triangle, shear_factor, 'y')
plot_objects(triangle, triangle_sheared_y, 'Triangle', f'Sheared Triangle (Y-axis, shear factor={shear_factor})')