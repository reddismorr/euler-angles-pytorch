import numpy as np
import torch
from typing import (
    Union, 
    List
)


def axis_rotation_matrix_from_angle(
    angles: Union[List, np.ndarray, torch.Tensor],
    axis: str,
    degrees: bool = True,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Create a rotation matrix (or matrices) for a given axis and set of angles.
    Supports both NumPy arrays and PyTorch tensors.

    Parameters
    ----------
    angles : list or np.ndarray or torch.Tensor
        Angles in degrees. Shape: () or (N,)
    axis : {'x', 'y', 'z'}
        Axis of rotation.

    Returns
    -------
    R : np.ndarray or torch.Tensor
        Rotation matrix or matrices of shape (3,3) or (N,3,3).
    """
    axis = axis.lower()
    if axis not in "xyz":
        raise ValueError("Invalid axis, must be 'x', 'y' or 'z'")
        
    if isinstance(angles, torch.Tensor):
        if degrees:
            angles = torch.deg2rad(angles)
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        eye = torch.eye(3, dtype=angles.dtype, device=angles.device)
        if angles.ndim > 0:
            R = eye.repeat(*angles.shape, 1, 1)
        else:
            R = eye
    else:
        angles = np.asarray(angles, dtype=np.float32)
        if degrees:
            angles = np.deg2rad(angles)
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        eye = np.eye(3, dtype=np.float32)
        if angles.ndim > 0:
            R = np.tile(eye, (*angles.shape, 1, 1))
        else:
            R = eye

    # Apply rotation logic (shared for both NumPy and Torch)
    if axis == 'x':
        R[..., 1, 1] = cos_a
        R[..., 1, 2] = -sin_a
        R[..., 2, 1] = sin_a
        R[..., 2, 2] = cos_a
    elif axis == 'y':
        R[..., 0, 0] = cos_a
        R[..., 0, 2] = sin_a
        R[..., 2, 0] = -sin_a
        R[..., 2, 2] = cos_a
    elif axis == 'z':
        R[..., 0, 0] = cos_a
        R[..., 0, 1] = -sin_a
        R[..., 1, 0] = sin_a
        R[..., 1, 1] = cos_a
    return R


def euler_angles_to_rotation_matrices(
    euler_angles: Union[List, np.ndarray, torch.Tensor],
    convention: str,
    intrinsic: bool = True,
    right_hand: bool = True,
    degrees: bool = True,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert Euler angles to rotation matrices, supporting NumPy and PyTorch.

    Parameters
    ----------
    euler_angles : (3,) or (n, 3) array-like
        The Euler angles (degrees or radians).
    convention : str
        A 3-character string (e.g. 'zyx', 'zxz').
    intrinsic : bool
        Use intrinsic rotation order (rotating frame).
    right_hand : bool
        If False, flips sign of all angles (left-hand system).
    degrees : bool
        If True, assumes input angles are in degrees.

    Returns
    -------
    R_total : (3, 3) or (n, 3, 3) ndarray or tensor
        The resulting rotation matrix (or matrices).
    """
    # Convert input lists to np.array
    if isinstance(euler_angles, list):
        euler_angles = np.asarray(euler_angles, dtype=np.float32)
    if euler_angles.shape[-1] != 3:
        raise ValueError("Invalid euler angles, must be 3 angles along last dimension")
    
    # convention check
    convention = convention.lower()
    if len(convention) != 3 or any(ax not in "xyz" for ax in convention):
        raise ValueError("Invalid convention, must be 3 chars from 'x', 'y', 'z'")
    
    # Carry right-hand rotations
    if not right_hand:
        euler_angles = -euler_angles

    # Apply rotations in order
    R_axis = [
        axis_rotation_matrix_from_angle(euler_angles[..., i], axis, degrees)
        for i, axis in enumerate(convention)
    ]
    
    # Carry intrinsic|extrinsic rotations
    if intrinsic:
        R_total = R_axis[0] @ R_axis[1] @ R_axis[2]
    else:
        R_total = R_axis[2] @ R_axis[1] @ R_axis[0]
    return R_total