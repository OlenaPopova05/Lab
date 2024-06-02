import numpy as np
import matplotlib.pyplot as plt
import cv2


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
    print(f"{title2}:\n {points2}")
    plt.show()


def plot_objects_3d(points1, points2, title1, title2):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121, projection='3d')
    ax.plot(points1[:, 0], points1[:, 1], points1[:, 2], marker='o')
    plt.title(title1)
    ax = fig.add_subplot(122, projection='3d')
    ax.plot(points2[:, 0], points2[:, 1], points2[:, 2], marker='o')
    plt.title(title2)
    print(f"{title2}:\n {points2}")
    plt.show()


def object_rotation(points, angle):
    angle = np.radians(angle)
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return np.dot(rotation_matrix, points.T).T


def object_rotation_cv(points, angle):
    rotation_matrix = cv2.getRotationMatrix2D((0, 0), angle, 1)
    return cv2.transform(points.reshape(-1, 1, 2), rotation_matrix).reshape(-1, 2)


def object_scaling(points, scale):
    scaling_matrix = np.array([[scale, 0], [0, scale]])
    return np.dot(points, scaling_matrix)


def object_scaling_cv(points, scale):
    scaling_matrix = np.array([[scale, 0], [0, scale]])
    return cv2.transform(points.reshape(-1, 1, 2), scaling_matrix).reshape(-1, 2)


def object_axis_reflection(points, axis):
    if axis == 'x':
        reflection_matrix = np.array([[1, 0], [0, -1]])
    else:
        reflection_matrix = np.array([[-1, 0], [0, 1]])
    return np.dot(points, reflection_matrix)


def object_axis_reflection_cv(points, axis):
    if axis == 'x':
        reflection_matrix = np.array([[1, 0], [0, -1]])
    else:
        reflection_matrix = np.array([[-1, 0], [0, 1]])
    return cv2.transform(points.reshape(-1, 1, 2), reflection_matrix).reshape(-1, 2)


def object_shear(points, shear_factor, axis):
    if axis == 'x':
        shear_matrix = np.array([[1, shear_factor], [0, 1]])
    else:
        shear_matrix = np.array([[1, 0], [shear_factor, 1]])
    return np.dot(shear_matrix, points.T).T


def object_shear_cv(points, shear_factor, axis):
    if axis == 'x':
        shear_matrix = np.array([[1, shear_factor], [0, 1]])
    else:
        shear_matrix = np.array([[1, 0], [shear_factor, 1]])
    return cv2.transform(points.reshape(-1, 1, 2), shear_matrix).reshape(-1, 2)


def custom_transformation(points, transformation_matrix):
    return np.dot(transformation_matrix, points.T).T


def custom_transformation_cv(points, transformation_matrix):
    return cv2.transform(points.reshape(-1, 1, 2), transformation_matrix).reshape(-1, 2)


pyramid = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 0],
    [0.5, 0.5, 1],
    [1, 0, 0],
    [1, 1, 0],
    [0.5, 0.5, 1],
    [0, 1, 0],
    [0, 0, 0],
    [0.5, 0.5, 1]
])

pyramid_custom_transformed_1 = custom_transformation(pyramid, np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]]))
plot_objects_3d(pyramid, pyramid_custom_transformed_1, 'Pyramid', 'Custom Transformed Pyramid 1')

angle_3d = np.radians(45)
pyramid_custom_transformed_2 = custom_transformation(pyramid, np.array([[np.cos(angle_3d), -np.sin(angle_3d), 0],
                                                                        [np.sin(angle_3d), np.cos(angle_3d), 0],
                                                                        [0, 0, 1]]))
plot_objects_3d(pyramid, pyramid_custom_transformed_2, 'Pyramid', 'Custom Transformed Pyramid 2')


triangle = np.array([[0, 0], [1, 0], [0.2, 1], [0, 0]])
trapezoid = np.array([[0, 0], [0, 1], [2, 1], [3, 0], [0, 0]])

trapezoid_rotated = object_rotation(trapezoid, 90)
plot_objects(trapezoid, trapezoid_rotated, 'Trapezoid', 'Rotated Trapezoid')

triangle_rotated = object_rotation(triangle, 90)
plot_objects(triangle, triangle_rotated, 'Triangle', 'Rotated Triangle')

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


custom_transformation_matrix = np.array([[2, 0], [0, 2]])

trapezoid_custom_transformed = custom_transformation(trapezoid, custom_transformation_matrix)
plot_objects(trapezoid, trapezoid_custom_transformed, 'Trapezoid', 'Custom Transformed Trapezoid')

triangle_custom_transformed = custom_transformation(triangle, custom_transformation_matrix)
plot_objects(triangle, triangle_custom_transformed, 'Triangle', 'Custom Transformed Triangle')




triangle_rotated_cv = object_rotation_cv(triangle, 90)
plot_objects(triangle, triangle_rotated_cv, 'Triangle', 'Rotated Triangle (OpenCV)')

triangle_scaled_cv = object_scaling_cv(triangle, scale_factor)
plot_objects(triangle, triangle_scaled_cv, 'Triangle', f'Scaled Triangle (scale factor={scale_factor}) (OpenCV)')

triangle_reflected_x_cv = object_axis_reflection_cv(triangle, 'x')
plot_objects(triangle, triangle_reflected_x_cv, 'Triangle', 'Reflected Triangle (X-axis) (OpenCV)')
triangle_reflected_y_cv = object_axis_reflection_cv(triangle, 'y')
plot_objects(triangle, triangle_reflected_y_cv, 'Triangle', 'Reflected Triangle (Y-axis) (OpenCV)')

triangle_sheared_x_cv = object_shear_cv(triangle, shear_factor, 'x')
plot_objects(triangle, triangle_sheared_x_cv, 'Triangle', f'Sheared Triangle (X-axis, shear factor={shear_factor}) (OpenCV)')
triangle_sheared_y_cv = object_shear_cv(triangle, shear_factor, 'y')
plot_objects(triangle, triangle_sheared_y_cv, 'Triangle', f'Sheared Triangle (Y-axis, shear factor={shear_factor}) (OpenCV)')


triangle_custom_transform_cv = custom_transformation_cv(triangle, custom_transformation_matrix)
plot_objects(triangle, triangle_custom_transform_cv, 'Triangle', 'Custom Transformed Triangle (OpenCV)')
