# euler-angles-pytorch

Simple and robust conversion between Euler angles and rotation matrices for NumPy and PyTorch.
Handles gimbal locks, supports intrinsic/extrinsic, right/left-handed rotations.


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


## Motivation


- PyTorch3D has known bugs with gimbal locks
- scipy.spatial doesn't support PyTorch
- euleangles doesn't support PyTorch
- This package works with both NumPy and Torch out of the box
