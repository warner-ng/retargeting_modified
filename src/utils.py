import torch
import torch.nn.functional as F
import numpy as np
import pinocchio as pin
'''主要使用pinnochio的接口实现了旋转、平移、变换、RollPitchYaw'''

def batch_rodrigues(rot_vecs): # 1. **batch_rodrigues**: 将批量轴角向量转换为旋转矩阵。
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3 array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''
    batch_size = rot_vecs.shape[0] # .shape[0] 获取张量的第一维大小，即批量尺寸
    device = rot_vecs.device
    dtype = rot_vecs.dtype

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True) # .norm() 计算张量的范数，这里用于计算旋转向量的幅度（角度）
                                                             #  + 1e-8 避免零值导致的数值不稳定。
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    flatten_K = [zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros]
    K = torch.cat(flatten_K, dim=1).reshape((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device)[None, ...]
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat
  
  
def transform_mat(R, t): # 2. **transform_mat**: 实现了将旋转矩阵 R 和平移向量 t 结合为齐次变换矩阵 T 的功能，用于刚体变换。
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)
  

def batch_rigid_transform(rot_mats, joints, parents): # 给joint做rotation变换
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    
    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    """
    joints = torch.unsqueeze(joints, dim=-1) # .unsqueeze() 在指定维度增加一个大小为 1 的维度，这里在最后一维增加，便于后续矩阵运算
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3), # 该操作将rot_mats重塑为形状(-1, 3, 3)，其中-1会自动推导大小。
                                    # 假设rot_mats原本是二维数组（例如形状为(n, 9)），它会被转换为三维数组
        rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]
    return transforms, posed_joints
  
# 矩阵2欧拉角
def mat2rpy(mat):
    rpy = pin.utils.matrixToRpy(mat.cpu().numpy())
    return torch.tensor(rpy, dtype=torch.float, device=mat.device)

# 欧拉角2矩阵
def rpy2mat(rpy):
    r, p, y = np.split(rpy.cpu().numpy(), 3, axis=-1)
    mat = pin.utils.rpyToMatrix(r[0].item(), p[0].item(), y[0].item())
    return torch.tensor(mat, dtype=torch.float, device=rpy.device)

# 四元数to欧拉角
def quat2rpy(q):
    q = normalize(q)
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[..., qw] * q[..., qx] + q[..., qy] * q[..., qz])
    cosr_cosp = q[..., qw] * q[..., qw] - q[..., qx] * \
                q[..., qx] - q[..., qy] * q[..., qy] + q[..., qz] * q[..., qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[..., qw] * q[..., qy] - q[..., qz] * q[..., qx])
    pitch = torch.where(
        torch.abs(sinp) >= 1, copysign(np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[..., qw] * q[..., qz] + q[..., qx] * q[..., qy])
    cosy_cosp = q[..., qw] * q[..., qw] + q[..., qx] * \
                q[..., qx] - q[..., qy] * q[..., qy] - q[..., qz] * q[..., qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return torch.stack((roll, pitch, yaw), dim=-1)

# 欧拉角to四元数
def rpy2quat(rpy):
    roll = rpy[..., 0] # ...表示任意数量的先行维度
    pitch = rpy[..., 1]
    yaw = rpy[..., 2]
    
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    q = torch.stack([qx, qy, qz, qw], dim=-1)
    return normalize(q)

# 递归调用前面的函数就行
def mat2quat(mat):
    return rpy2quat(mat2rpy(mat))


def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)

# b的符号给a
def copysign(a, b):
    a = torch.zeros_like(b) + a  # 用+a确保a的原始数据类型和属性不变
    return torch.abs(a) * torch.sign(b)