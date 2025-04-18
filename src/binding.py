import utils
import torch
import numpy as np
import pinocchio as pin
from pinocchio import BODY

# SMPL（Skinned Multi-Person Linear Model）是一种用于人体建模的参数化模型
smpl_body_names = [
    'pelvis',
    'r_hip', 'l_hip',
    'spine1',
    'r_knee', 'l_knee',
    'spine2',
    'r_ankle', 'l_ankle',
    'spine3',
    'r_foot', 'l_foot',
    'neck',
    'r_collar', 'l_collar',
    'head',
    'r_shoulder', 'l_shoulder',
    'r_elbow', 'l_elbow',
    'r_wrist', 'l_wrist',
]
# 关节校准（Joint Calibration）
# ???不懂如何进行的joint_calibration 校准位置???  不懂的主要点是：哪些是retargeting基本的数据结构（类）；
def J_calibration(joints, robot, t_pose_q, parents, bindings, hackings):
    pin.computeJointJacobians(robot.model, robot.data, t_pose_q)
    pin.updateFramePlacements(robot.model, robot.data)
        
    dtype, device, batch_size = joints.dtype, joints.device, joints.shape[0]
    axes_rot = utils.rpy2mat(torch.tensor([0., -np.pi/2, -np.pi/2], device=device, dtype=dtype))
    
    rel_joints = joints.clone() @ axes_rot
    rel_joints[:, 1:] -= joints[:, parents[1:]] @ axes_rot
    
    chain = [rel_joints[:, 0]]
    for i in range(1, parents.shape[0]):
        curr_res = chain[parents[i]] + rel_joints[:, i]
        chain.append(curr_res)
    smpl_joint_pos = torch.stack(chain, dim=1)
    
    # Calibration stages:
    # 1. Upper bodies (w.r.t collar)
    #    Point-to-point bindings
    #    (1). 
    # 2. Lower bodies (w.r.t hip)
    #    (1). Hip position translation
    #    (2). Leg ratio scale (z-axis)
    tensor = lambda x: torch.tensor(x, dtype=dtype, device=device)[None].repeat(batch_size, 1)
    get_smpl_id = lambda n: smpl_body_names.index(n)
    get_body_pos = lambda n: robot.data.oMf[robot.model.getFrameId(n, BODY)].np[:3, 3]
    get_body_pos_wrt_root = lambda n, root: get_body_pos(n) - get_body_pos(root) # wrt: with respect to相对位置
    get_parent_name = lambda n: smpl_body_names[parents[get_smpl_id(n)]]
    get_joint_pos_wrt_root = lambda n, root: smpl_joint_pos[:, get_smpl_id(n)] - smpl_joint_pos[:, get_smpl_id(root)]
    
    # Leg calibration
    rel_joints[:, 2] = tensor(get_body_pos_wrt_root(bindings['l_hip'], bindings[get_parent_name('l_hip')]))
    rel_joints[:, 1] = tensor(get_body_pos_wrt_root(bindings['r_hip'], bindings[get_parent_name('r_hip')]))
    
    robot_l_leg_height = tensor(get_body_pos_wrt_root(bindings['l_ankle'], bindings['l_hip']))[:, 2]
    smpl_l_leg_height = get_joint_pos_wrt_root('l_ankle', 'l_hip')[:, 2]
    robot_l_leg_ratio = robot_l_leg_height / smpl_l_leg_height
    rel_joints[:, [5, 8]] *= robot_l_leg_ratio[:, None, None]
    
    robot_r_leg_height = tensor(get_body_pos_wrt_root(bindings['r_ankle'], bindings['r_hip']))[:, 2]
    smpl_r_leg_height = get_joint_pos_wrt_root('r_ankle', 'r_hip')[:, 2]
    robot_r_leg_ratio = robot_r_leg_height  / smpl_r_leg_height
    rel_joints[:, [4, 7]] *= robot_r_leg_ratio[:, None, None]
    
    # Arm calibration
    rel_joints[:, 17] = tensor(get_body_pos_wrt_root(bindings['l_shoulder'], bindings[get_parent_name('l_shoulder')]))
    rel_joints[:, 16] = tensor(get_body_pos_wrt_root(bindings['r_shoulder'], bindings[get_parent_name('r_shoulder')]))
    rel_joints[:, 19] = tensor(get_body_pos_wrt_root(bindings['l_elbow'], bindings[get_parent_name('l_elbow')]))
    rel_joints[:, 18] = tensor(get_body_pos_wrt_root(bindings['r_elbow'], bindings[get_parent_name('r_elbow')]))
    rel_joints[:, 21] = tensor(get_body_pos_wrt_root(bindings['l_wrist'], bindings[get_parent_name('l_wrist')]))
    rel_joints[:, 20] = tensor(get_body_pos_wrt_root(bindings['r_wrist'], bindings[get_parent_name('r_wrist')]))
    
    # Head calibration
    rel_joints[:, 3] = tensor(get_body_pos_wrt_root(bindings['spine1'], bindings['pelvis']))
    rel_joints[:, 9] = tensor(get_body_pos_wrt_root(bindings['spine3'], bindings['spine1']))
    rel_joints[:, 6] = 0.0
    
    chain = [rel_joints[:, 0]]
    for i in range(1, parents.shape[0]):
        curr_res = chain[parents[i]] + rel_joints[:, i]
        chain.append(curr_res)
    calibrated_joint_pos = torch.stack(chain, dim=1)
    
    for hacking, vec in hackings.items():
        calibrated_joint_pos[:, get_smpl_id(hacking)] += tensor(vec)
        
    # Lower body height ratio
    robot_lower_body_height = tensor(get_body_pos_wrt_root(bindings['l_ankle'], bindings['pelvis']))[:, 2] + \
        tensor(get_body_pos_wrt_root(bindings['r_ankle'], bindings['pelvis']))[:, 2]
    smpl_lower_body_height = get_joint_pos_wrt_root('l_ankle', 'pelvis')[:, 2] + get_joint_pos_wrt_root('r_ankle', 'pelvis')[:, 2]
    
    joints = calibrated_joint_pos @ torch.inverse(axes_rot)
    ratio = robot_lower_body_height.abs() / smpl_lower_body_height.abs() # robot和smpl的比例
    return joints, ratio