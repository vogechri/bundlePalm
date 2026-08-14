import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def skew(vector):
    x_value, y_value, z_value = vector
    return np.array(
        [
            [0.0, -z_value, y_value],
            [z_value, 0.0, -x_value],
            [-y_value, x_value, 0.0],
        ]
    )


def so3_left_jacobian(rotation):
    angle = np.linalg.norm(rotation)
    cross = skew(rotation)
    if angle < 1e-10:
        return np.eye(3) + 0.5 * cross + cross @ cross / 6.0
    return (
        np.eye(3)
        + (1.0 - np.cos(angle)) / angle**2 * cross
        + (angle - np.sin(angle)) / angle**3 * cross @ cross
    )


def camera_plus(camera, tangent, mode):
    current_rotation = Rotation.from_rotvec(camera[:3])
    rotation_increment = Rotation.from_rotvec(tangent[3:6])
    updated = camera.copy()
    if mode == "se3_right":
        updated[:3] = (current_rotation * rotation_increment).as_rotvec()
        updated[3:6] += current_rotation.apply(
            so3_left_jacobian(tangent[3:6]) @ tangent[:3]
        )
    else:
        updated[:3] = (rotation_increment * current_rotation).as_rotvec()
        if mode == "se3_left":
            updated[3:6] = rotation_increment.apply(camera[3:6]) + (
                so3_left_jacobian(tangent[3:6]) @ tangent[:3]
            )
        elif mode == "so3_center_left":
            camera_center = -current_rotation.inv().apply(camera[3:6])
            updated_center = camera_center + tangent[:3]
            updated[3:6] = -(
                rotation_increment * current_rotation
            ).apply(updated_center)
        else:
            updated[3:6] += tangent[:3]
    updated[6:9] += tangent[6:9]
    return updated


def analytic_camera_jacobian(camera, mode):
    result = np.zeros((9, 9))
    rotation = camera[:3]
    if mode == "se3_right":
        result[:3, 3:6] = np.linalg.inv(so3_left_jacobian(-rotation))
        result[3:6, :3] = Rotation.from_rotvec(rotation).as_matrix()
    else:
        result[:3, 3:6] = np.linalg.inv(so3_left_jacobian(rotation))
        if mode == "so3_center_left":
            result[3:6, :3] = -Rotation.from_rotvec(rotation).as_matrix()
            result[3:6, 3:6] = -skew(camera[3:6])
        else:
            result[3:6, :3] = np.eye(3)
        if mode == "se3_left":
            result[3:6, 3:6] = -skew(camera[3:6])
    result[6:9, 6:9] = np.eye(3)
    return result


@pytest.mark.parametrize(
    "mode", ["so3_left", "so3_center_left", "se3_left", "se3_right"]
)
def test_camera_tangent_jacobian_matches_finite_difference(mode):
    camera = np.array(
        [0.31, -0.22, 0.17, 1.2, -0.7, 2.1, 800.0, 0.01, -0.001]
    )
    epsilon = 1e-7
    numerical = np.empty((9, 9))
    for column in range(9):
        tangent = np.zeros(9)
        tangent[column] = epsilon
        positive = camera_plus(camera, tangent, mode)
        tangent[column] = -epsilon
        negative = camera_plus(camera, tangent, mode)
        numerical[:, column] = (positive - negative) / (2.0 * epsilon)

    np.testing.assert_allclose(
        numerical,
        analytic_camera_jacobian(camera, mode),
        rtol=2e-7,
        atol=5e-7,
    )


@pytest.mark.parametrize(
    "mode", ["so3_left", "so3_center_left", "se3_left", "se3_right"]
)
def test_point_action_tangent_jacobian_matches_finite_difference(mode):
    camera = np.array(
        [0.31, -0.22, 0.17, 1.2, -0.7, 2.1, 800.0, 0.01, -0.001]
    )
    point = np.array([0.4, -1.1, 3.2])
    rotation = Rotation.from_rotvec(camera[:3]).as_matrix()
    camera_point = rotation @ point + camera[3:6]
    if mode == "se3_left":
        analytic = np.column_stack((np.eye(3), -skew(camera_point)))
    elif mode == "se3_right":
        analytic = np.column_stack((rotation, -rotation @ skew(point)))
    elif mode == "so3_center_left":
        analytic = np.column_stack((-rotation, -skew(camera_point)))
    else:
        analytic = np.column_stack((np.eye(3), -skew(rotation @ point)))

    epsilon = 1e-7
    numerical = np.empty((3, 6))
    for column in range(6):
        tangent = np.zeros(9)
        tangent[column] = epsilon
        positive_camera = camera_plus(camera, tangent, mode)
        tangent[column] = -epsilon
        negative_camera = camera_plus(camera, tangent, mode)
        positive = (
            Rotation.from_rotvec(positive_camera[:3]).apply(point)
            + positive_camera[3:6]
        )
        negative = (
            Rotation.from_rotvec(negative_camera[:3]).apply(point)
            + negative_camera[3:6]
        )
        numerical[:, column] = (positive - negative) / (2.0 * epsilon)

    np.testing.assert_allclose(numerical, analytic, rtol=2e-7, atol=5e-7)


def test_camera_center_product_updates_center_independently():
    camera = np.array(
        [0.31, -0.22, 0.17, 1.2, -0.7, 2.1, 800.0, 0.01, -0.001]
    )
    tangent = np.array(
        [0.4, -0.3, 0.2, 0.08, -0.04, 0.03, 0.0, 0.0, 0.0]
    )
    original_rotation = Rotation.from_rotvec(camera[:3])
    original_center = -original_rotation.inv().apply(camera[3:6])

    updated = camera_plus(camera, tangent, "so3_center_left")
    updated_rotation = Rotation.from_rotvec(updated[:3])
    updated_center = -updated_rotation.inv().apply(updated[3:6])

    np.testing.assert_allclose(updated_center, original_center + tangent[:3])


def test_product_subspace_metric_congruence_is_coherent():
    random = np.random.default_rng(17)
    tangent_to_stored = random.normal(size=(9, 9))
    tangent_to_stored += 4.0 * np.eye(9)
    factor = random.normal(size=(9, 9))
    stored_metric = factor.T @ factor + np.eye(9)
    ratio = 2.5
    subspace_scale = np.eye(9)
    subspace_scale[:3, :3] *= np.sqrt(ratio)

    tangent_metric = tangent_to_stored.T @ stored_metric @ tangent_to_stored
    inverse_tangent = np.linalg.inv(tangent_to_stored)
    adjusted_stored = (
        inverse_tangent.T
        @ subspace_scale
        @ tangent_metric
        @ subspace_scale
        @ inverse_tangent
    )
    recovered_tangent = (
        tangent_to_stored.T @ adjusted_stored @ tangent_to_stored
    )

    np.testing.assert_allclose(
        recovered_tangent,
        subspace_scale @ tangent_metric @ subspace_scale,
        rtol=1e-11,
        atol=1e-10,
    )
    np.testing.assert_allclose(adjusted_stored, adjusted_stored.T, atol=1e-10)