# euler-angles-pytorch

Simple and robust conversion between Euler angles and rotation matrices for NumPy and PyTorch.
Handles gimbal locks, supports intrinsic/extrinsic, right/left-handed rotations.

## Motivation

- PyTorch3D has known bugs with gimbal locks
- scipy.spatial doesn't support PyTorch
- euleangles doesn't support PyTorch
- This package works with both NumPy and Torch out of the box

## Install

```bash
pip install git+https://github.com/reddismorr/euler-angles-pytorch.git
```

## Usage

Just a simple example of translating angles from one convention to another: 

```python
from euler_angles_pytorch import (
	euler_angles_to_rotation_matrices,
	rotation_matrices_to_euler_angles
)

R = euler_angles_to_rotation_matrices(
    [
        [0, 0, 90],
        [0, 90, 0],
    ], 
    convention='zyz', 
    intrinsic=True,
    right_hand=True,
    degrees=True,
)
```
```python
>>> R
array([[[-4.371139e-08, -1.000000e+00,  0.000000e+00],
        [ 1.000000e+00, -4.371139e-08,  0.000000e+00],
        [ 0.000000e+00,  0.000000e+00,  1.000000e+00]],

       [[-4.371139e-08,  0.000000e+00,  1.000000e+00],
        [ 0.000000e+00,  1.000000e+00,  0.000000e+00],
        [-1.000000e+00,  0.000000e+00, -4.371139e-08]]], dtype=float32)
```
```python
angles = rotation_matrices_to_euler_angles(
        R, 
        convention='xyx', 
        intrinsic=True,
        right_hand=True,
        degrees=True,
    )
```
```python
>>> angles
array([[ 90.,  90., -90.],
       [  0.,  90.,   0.]], dtype=float32)
```

## Little explanation for curious ones

There is a well-defined but somewhat tricky scheme for recovering Euler angles `(alpha, beta, gamma)` from a rotation matrix `R`. The key part of this process is the second angle `(beta)`, since it determines whether the rotation is near a gimbal lock condition.

In both intrinsic and extrinsic rotations, the second angle is central and governs the interaction between the first and last rotation axes.

### Step 1: Extract the second angle
 
This angle usually appears as either `sin(beta)` or `cos(beta)` in some position inside the matrix `R`. If you know:

- which matrix element it appears in,

- whether to apply arcsin or arccos,

- what the sign it has,

then extracting beta is trivial.

### Step 2: Detect gimbal lock

There are two types of gimbal lock:
1. Same-axis schemes (like 'xyx', 'zxz', etc.):
    Gimbal lock happens when `beta ≈ 0` (i.e., `sin(beta) ≈ 0`), because the first and last axes coincide. Only the sum of the first and last angles is meaningful in that case.
    A common strategy is to set the third angle `gamma = 0` and solve for `alpha`.
2. All-different-axis schemes (like 'zyx', 'yxz', etc.):
    Gimbal lock happens when `beta ≈ ±90°` (i.e., `cos(beta) ≈ 0`), because the third axis becomes aligned with the first one. Again, one of the outer angles is fixed, and the other is computed accordingly.

In both cases, the fallback is usually based on some `arctan2` expression involving matrix elements. These are hardcoded per-notation in this implementation.

## Implementation notes
The logic here is consistent with [eulerangles](https://github.com/alisterburt/eulerangles) by [Alister Burt](https://github.com/alisterburt), but this implementation is lighter (for my hunble optimion of course).

It supports both NumPy and PyTorch inputs and outputs.

### pytorch3d problem
pytorch3d's matrix_to_euler_angles appears to be buggy. For example, ZYZ with angles `(0, 0, 45)` produces `(0, 0, 180)` as inverse, which is technically a valid solution but not the expected one for tomography applications.

Try this for comparison:
```python 
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles
import torch

angles = torch.rand(3)
angles[1] = 0.0
# angles = torch.tensor([3, 0, 45.0])
angles_rad = torch.deg2rad(angles)
R = euler_angles_to_matrix(angles_rad, 'ZYZ')
angles_back = torch.rad2deg(matrix_to_euler_angles(R, 'ZYZ'))

print(f'Initial angles : {angles.numpy()}')
print(f'Rotation matrix :')
print(R.numpy())
print(f'Angles after conversion : {angles_back.numpy()}')

```
You should see something like this:
```python
Initial angles : [0.12240285 0.         0.32032895]
Rotation matrix :
[[ 0.9999702  -0.00772705  0.        ]
 [ 0.00772705  0.9999702   0.        ]
 [ 0.          0.          1.        ]]
Angles after conversion : [  0.   0. 180.]
```
It comes from how they calculate the first and the last angle with `atan2` from zeros. 

