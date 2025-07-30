import numpy as np
import torch
from typing import (
    Union, 
    List
)

_valid_conventions = {
    'xyx', 'xyz', 'xzx', 'xzy',
    'yxy', 'yxz', 'yzx', 'yzy',
    'zxy', 'zxz', 'zyx', 'zyz'
}

_extrinsic_schemes = {
    'xyx': {
        'angle1': ('arccos', +1, (0, 0)),
        'gimbal_check': (0, 0, 1.0),
        'gimbal_case': {
            'angle0': ('arctan2', (-1, (1, 2)), ( +1, (1, 1))),
            'angle2': 0.0,
        },
        'normal_case': {
            'angle0': ('arctan2', ( +1, (0, 1)), ( +1, (0, 2))),
            'angle2': ('arctan2', ( +1, (1, 0)), (-1, (2, 0))),
        }
    },
    'xyz': {
        'angle1': ('arcsin', -1, (2, 0)),
        'gimbal_check': (2, 0, -1.0),
        'gimbal_case': {
            'angle0': ('arctan2', (-1, (1, 2)), ( +1, (1, 1))),
            'angle2': 0.0,
        },
        'normal_case': {
            'angle0': ('arctan2', ( +1, (2, 1)), ( +1, (2, 2))),
            'angle2': ('arctan2', ( +1, (1, 0)), ( +1, (0, 0))),
        }
    },
    'xzx': {
        'angle1': ('arccos', +1, (0, 0)),
        'gimbal_check': (0, 0, 1.0),
        'gimbal_case': {
            'angle0': ('arctan2', ( +1, (2, 1)), ( +1, (2, 2))),
            'angle2': 0.0,
        },
        'normal_case': {
            'angle0': ('arctan2', ( +1, (0, 2)), (-1, (0, 1))),
            'angle2': ('arctan2', ( +1, (2, 0)), ( +1, (1, 0))),
        }
    },
    'xzy': {
        'angle1': ('arcsin', +1, (1, 0)),
        'gimbal_check': (1, 0, 1.0),
        'gimbal_case': {
            'angle0': ('arctan2', ( +1, (2, 1)), ( +1, (2, 2))),
            'angle2': 0.0,
        },
        'normal_case': {
            'angle0': ('arctan2', (-1, (1, 2)), ( +1, (1, 1))),
            'angle2': ('arctan2', (-1, (2, 0)), ( +1, (0, 0))),
        }
    },
    'yxy': {
        'angle1': ('arccos', +1, (1, 1)),
        'gimbal_check': (1, 1, 1.0),
        'gimbal_case': {
            'angle0': ('arctan2', ( +1, (0, 2)), ( +1, (0, 0))),
            'angle2': 0.0,
        },
        'normal_case': {
            'angle0': ('arctan2', ( +1, (1, 0)), (-1, (1, 2))),
            'angle2': ('arctan2', ( +1, (0, 1)), ( +1, (2, 1))),
        }
    },
    'yxz': {
        'angle1': ('arcsin', +1, (2, 1)),
        'gimbal_check': (2, 1, 1.0),
        'gimbal_case': {
            'angle0': ('arctan2', ( +1, (0, 2)), ( +1, (0, 0))),
            'angle2': 0.0,
        },
        'normal_case': {
            'angle0': ('arctan2', (-1, (2, 0)), ( +1, (2, 2))),
            'angle2': ('arctan2', (-1, (0, 1)), ( +1, (1, 1))),
        }
    },
    'yzx': {
        'angle1': ('arcsin', -1, (0, 1)),
        'gimbal_check': (0, 1, -1.0),
        'gimbal_case': {
            'angle0': ('arctan2', (-1, (2, 0)), ( +1, (2, 2))),
            'angle2': 0.0,
        },
        'normal_case': {
            'angle0': ('arctan2', ( +1, (0, 2)), ( +1, (0, 0))),
            'angle2': ('arctan2', ( +1, (2, 1)), ( +1, (1, 1))),
        }
    },
    'yzy': {
        'angle1': ('arccos', +1, (1, 1)),
        'gimbal_check': (1, 1, 1.0),
        'gimbal_case': {
            'angle0': ('arctan2', (-1, (2, 0)), ( +1, (2, 2))),
            'angle2': 0.0,
        },
        'normal_case': {
            'angle0': ('arctan2', ( +1, (1, 2)), ( +1, (1, 0))),
            'angle2': ('arctan2', ( +1, (2, 1)), (-1, (0, 1))),
        }
    },
    'zxy': {
        'angle1': ('arcsin', -1, (1, 2)),
        'gimbal_check': (1, 2, -1.0),
        'gimbal_case': {
            'angle0': ('arctan2', (-1, (0, 1)), ( +1, (0, 0))),
            'angle2': 0.0,
        },
        'normal_case': {
            'angle0': ('arctan2', ( +1, (1, 0)), ( +1, (1, 1))),
            'angle2': ('arctan2', ( +1, (0, 2)), ( +1, (2, 2))),
        }
    },
    'zxz': {
        'angle1': ('arccos', +1, (2, 2)),
        'gimbal_check': (2, 2, 1.0),
        'gimbal_case': {
            'angle0': ('arctan2', (-1, (0, 1)), ( +1, (0, 0))),
            'angle2': 0.0,
        },
        'normal_case': {
            'angle0': ('arctan2', ( +1, (2, 0)), ( +1, (2, 1))),
            'angle2': ('arctan2', ( +1, (0, 2)), (-1, (1, 2))),
        }
    },
    'zyx': {
        'angle1': ('arcsin', +1, (0, 2)),
        'gimbal_check': (0, 2, 1.0),
        'gimbal_case': {
            'angle0': ('arctan2', ( +1, (1, 0)), ( +1, (1, 1))),
            'angle2': 0.0,
        },
        'normal_case': {
            'angle0': ('arctan2', (-1, (0, 1)), ( +1, (0, 0))),
            'angle2': ('arctan2', (-1, (1, 2)), ( +1, (2, 2))),
        }
    },
    'zyz': {
        'angle1': ('arccos', +1, (2, 2)),
        'gimbal_check': (2, 2, 1.0),
        'gimbal_case': {
            'angle0': ('arctan2', ( +1, (1, 0)), ( +1, (1, 1))),
            'angle2': 0.0,
        },
        'normal_case': {
            'angle0': ('arctan2', ( +1, (2, 1)), (-1, (2, 0))),
            'angle2': ('arctan2', ( +1, (1, 2)), ( +1, (0, 2))),
        }
    },
}


def _gimbal_check(R, conditions, tolerance):
    if isinstance(R, torch.Tensor):
        absolute = torch.abs
    else:
        absolute = np.abs
    i, j, val = conditions
    return (absolute(R[..., i, j] - val) < tolerance)
    

def _compute_middle_euler_angle_from_matrix(scheme_dict, R):
    func_name, sign, (i, j) = scheme_dict['angle1']
    val = R[:, i, j]
    if func_name == 'arcsin':
        func = torch.arcsin if isinstance(R, torch.Tensor) else np.arcsin
    elif func_name == 'arccos':
        func = torch.arccos if isinstance(R, torch.Tensor) else np.arccos
    else:
        raise ValueError(f"Unknown function {func_name} in angle1 spec.")
    return sign * func(val)


def _compute_sise_angle_from_matrix(spec, R, idx_mask):
    if isinstance(R, torch.Tensor):
        out = torch.zeros(R.shape[:-2], dtype=R.dtype, device=R.device)
        atan2 = torch.atan2
    else:
        R = np.asarray(R, dtype=np.float32)
        out = np.zeros(R.shape[:-2], dtype=np.float32)
        atan2 = np.arctan2

    if not isinstance(spec, tuple):
        out[idx_mask] = spec
        return out

    f_name, (sgn1, (i1, j1)), (sgn2, (i2, j2)) = spec
    if f_name != 'arctan2':
        raise ValueError("Only 'arctan2' supported in the spec so far.")
    
    out[idx_mask] = atan2(
        sgn1 * R[idx_mask, i1, j1],
        sgn2 * R[idx_mask, i2, j2]
    )
    return out


def rotation_matrices_to_euler_angles(
    rotation_matrices: Union[List, np.ndarray, torch.Tensor],
    convention: str,
    intrinsic: bool = True,
    right_hand: bool = True,
    degrees: bool = True,
    tolerance: float = 1e-6,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert rotation matrices to Euler angles according to specified convention.

    Supports both NumPy and PyTorch input. Handles gimbal lock conditions explicitly
    using per-convention analytic rules. Second Euler angle (beta) is extracted directly;
    first and third are resolved either normally or via gimbal-lock-specific fallback.

    Parameters
    ----------
    rotation_matrices : list or np.ndarray or torch.Tensor
        One or more 3×3 rotation matrices. Shape (..., 3, 3).
    convention : str
        Three-letter Euler convention using axes 'x', 'y', 'z' (e.g., 'zyz', 'xyz').
        Must match a known scheme.
    intrinsic : bool, default=True
        If True, interpret as intrinsic rotations (rotating frame). Otherwise, extrinsic.
    right_hand : bool, default=True
        If False, result angles are inverted (left-handed convention).
    degrees : bool, default=True
        If True, return angles in degrees. Otherwise, in radians.
    tolerance : float, default=1e-6
        Threshold to detect gimbal lock based on critical matrix element.

    Returns
    -------
    angles : np.ndarray or torch.Tensor
        Euler angles with shape (..., 3), matching input batch shape.
        Order of angles matches the specified convention (or its inverse if intrinsic).
    """

    R = rotation_matrices
    if isinstance(R, torch.Tensor):
        angles = torch.zeros((*R.shape[:-2], 3), dtype=R.dtype, device=R.device)
        rad2deg = torch.rad2deg
    else:
        R = np.asarray(R, dtype=np.float32)
        angles = np.zeros((*R.shape[:-2], 3), dtype=np.float32)
        rad2deg = np.rad2deg
    
    convention = convention.lower()
    if convention not in _valid_conventions:
        raise ValueError(f"Invalid convention '{convention}'")
    if intrinsic:
        convention = convention[::-1]
    scheme_dict = _extrinsic_schemes[convention]
    
    angles[..., 1] = _compute_middle_euler_angle_from_matrix(scheme_dict, R)

    idx_gimbal = _gimbal_check(R, scheme_dict['gimbal_check'], tolerance)
    idx_normal = ~idx_gimbal

    spec0_gimbal = scheme_dict['gimbal_case']['angle0']
    spec2_gimbal = scheme_dict['gimbal_case']['angle2']
    angles[idx_gimbal, 0] = _compute_sise_angle_from_matrix(spec0_gimbal, R, idx_gimbal)[idx_gimbal]
    angles[idx_gimbal, 2] = _compute_sise_angle_from_matrix(spec2_gimbal, R, idx_gimbal)[idx_gimbal]

    spec0_normal = scheme_dict['normal_case']['angle0']
    spec2_normal = scheme_dict['normal_case']['angle2']
    angles[idx_normal, 0] = _compute_sise_angle_from_matrix(spec0_normal, R, idx_normal)[idx_normal]
    angles[idx_normal, 2] = _compute_sise_angle_from_matrix(spec2_normal, R, idx_normal)[idx_normal]

    if intrinsic:
        angles = angles[..., [2, 1, 0]]
    if not right_hand:
        angles *= -1
    if degrees:
        angles = rad2deg(angles)

    return angles