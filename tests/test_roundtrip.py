import pytest
import torch
import numpy as np

from euler_angles_pytorch import (
    euler_angles_to_rotation_matrices,
    rotation_matrices_to_euler_angles,
)

valid_conventions = {
    'xyx', 'xyz', 'xzx', 'xzy',
    'yxy', 'yxz', 'yzx', 'yzy',
    'zxy', 'zxz', 'zyx', 'zyz'
}

angles_to_test = [
    [0, 0, 0],
    [0, 90, 0],
    [0, 0, 90],
    [45, 60, 30],
    [-45, -60, -30],
    [180, 0, 180],
    [0, 180, 0],
    [0, 90, 90],
    [12, 0, 34],
    [90, 90, 90],
]

@pytest.mark.parametrize("convention", sorted(valid_conventions))
def test_rotation_roundtrip(convention):
    intrinsic = True
    right_hand = True
    degrees = True

    angles_in = torch.tensor(angles_to_test, dtype=torch.float32)
    
    R = euler_angles_to_rotation_matrices(
        angles_in,
        convention=convention,
        intrinsic=intrinsic,
        right_hand=right_hand,
        degrees=degrees,
    )
    
    angles_out = rotation_matrices_to_euler_angles(
        R,
        convention=convention,
        intrinsic=intrinsic,
        right_hand=right_hand,
        degrees=degrees,
    )

    R_again = euler_angles_to_rotation_matrices(
        angles_out,
        convention=convention,
        intrinsic=intrinsic,
        right_hand=right_hand,
        degrees=degrees,
    )

    angle_diff = torch.abs(angles_in - angles_out)
    matrix_diff = torch.abs(R - R_again)

    assert torch.all(matrix_diff < 1e-3), f"Matrix mismatch in {convention}"