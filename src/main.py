import warnings
warnings.filterwarnings("ignore")

import os
import pink
import yaml
import torch
import pickle
import argparse
import qpsolvers
import numpy as np
import pinocchio as pin

from tqdm import tqdm
from copy import deepcopy
from pink import solve_ik
from pinocchio import BODY
from loop_rate_limiters import RateLimiter
from pinocchio.visualize import MeshcatVisualizer
from pink.tasks import FrameTask, RelativeFrameTask, ComTask, LowAccelerationTask

import meshcat_shapes
import binding
import utils

MIN_HEIGHT_THRESHOLD = 0.02


smpl_body_names = [
    "pelvis",
    "r_hip", "l_hip",
    "spine1",
    "r_knee", "l_knee",
    "spine2",
    "r_ankle", "l_ankle",
    "spine3",
    "r_foot", "l_foot",
    "neck",
    "r_collar", "l_collar",
    "head",
    "r_shoulder", "l_shoulder",
    "r_elbow", "l_elbow",
    "r_wrist", "l_wrist",
]


human_ordered_joint_names = [
    "r_hip", "r_knee", "r_ankle", "r_foot",
    "l_hip", "l_knee", "l_ankle", "l_foot",
    "spine1", "spine2", "spine3", "neck", "head",
    "r_collar", "r_shoulder", "r_elbow", "r_wrist",
    "l_collar", "l_shoulder", "l_elbow", "l_wrist",
]

# 这些匿名函数（lambda表达式）用于定义简洁的、一次性的小型函数，通常包含参数和一个返回表达式
# 语法： lambda 参数1, 参数2, ... : 表达式，定义一个匿名函数，参数逗号分隔，表达式返回结果
get_smpl_id = lambda n: smpl_body_names.index(n)                                            # 根据SMPL模型的关节名称返回其索引
transform_wrt_world = lambda config, body: config.get_transform_frame_to_world(body).copy() # 获取某个物体相对于世界坐标系的变换矩阵。
tensors2se3 = lambda mat, vec: pin.SE3(mat.numpy(), vec.numpy())                            # 将旋转矩阵和平移向量转换为SE3变换
get_key_from_value = lambda d, v: list(d.keys())[list(d.values()).index(v)]                 # 根据字典值查找对应的键
transfrom2q = lambda x: torch.cat([x[:3, 3], utils.mat2quat(x[:3, :3])], dim=0).numpy()     # 将变换矩阵转换为四元数表示
transfrom2v = lambda x, y, t: torch.cat([(x[:3, 3] - y[:3, 3]) / t,                         # 计算两个变换之间的速度
    (utils.mat2rpy(x[:3, :3]) - utils.mat2rpy(y[:3, :3])) / t], dim=0).numpy()


def build_human_wrapper(path, filename="human.xml", root_joint=None):# 输出：封装好的人体模型对象
    model, collision_model, visual_model = pin.shortcuts.buildModelsFromMJCF(
        os.path.join(path, filename), root_joint=root_joint)
    human = pin.RobotWrapper(model=model, collision_model=collision_model, visual_model=visual_model)
    return human


def get_root_joint_dim(model):# 输出：根关节的平移和旋转自由度。
    if model.existJointName("root_joint"):
        root_joint_id = model.getJointId("root_joint")
        root_joint = model.joints[root_joint_id]
        return root_joint.nq, root_joint.nv
    return 0, 0
 

def load_smpl_models(folder):# 加载SMPL模型参数
    smpl_models = {}
    for gender in ["female", "male", "neutral"]:
        print(folder)
        print(gender)   
        smpl_model = pickle.load(open(os.path.join(folder, f"{gender}.pkl"),"rb"))
        v_template = torch.tensor(smpl_model["v_template"][None, ...], dtype=torch.float)
        shapedirs = torch.tensor(smpl_model["shapedirs"][:, :, :10], dtype=torch.float)
        try:
            J_regressor = torch.tensor(np.array(smpl_model["J_regressor"]), dtype=torch.float)
        except:
            J_regressor = torch.tensor(smpl_model["J_regressor"].toarray(), dtype=torch.float)
        parents = torch.tensor(smpl_model["kintree_table"][0].astype(np.int64), dtype=torch.long)
        smpl_models[gender] = (v_template, shapedirs, J_regressor[:22], parents[:22]) # 这是重点 J_regressor'：关节回归矩阵
    return smpl_models


def make_joint_config(args, robot, config, min_dict, max_dict, limit=0.95):# 从config里设置关节配置（最大最小角度限制）（config在pink里）
    q_max = config.model.upperPositionLimit.copy()
    q_min = config.model.lowerPositionLimit.copy()
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # Write joint ids mapping file
    with open(f"{args.output}/joint_id.txt", "w") as f:
        for joint_name in robot.model.names:
            joint_id = robot.model.getJointId(joint_name) - 2
            if joint_id >= 0:
                f.write(str(joint_id) + " " + joint_name + "\n")
                if args.print_joints:
                    print(joint_id, joint_name)
    
    start, _ = get_root_joint_dim(config.model)
    end = config.model.nq
    
    q_mean = (q_max[start:end] + q_min[start:end]) * 0.5
    q_scale = (q_max[start:end] - q_min[start:end]) * limit
    q_max[start:end] = q_mean + q_scale * 0.5
    q_min[start:end] = q_mean - q_scale * 0.5
    
    for joint_name in robot.model.names:
        joint_id = robot.model.getJointId(joint_name) - 2 + start
        if joint_name in min_dict.keys():
            q_min[joint_id] = max(q_min[joint_id], np.deg2rad(min_dict[joint_name]))
        if joint_name in max_dict.keys():
            q_max[joint_id] = min(q_max[joint_id], np.deg2rad(max_dict[joint_name]))
    return q_max, q_min


def qpos_clamp(config, q_max, q_min, root_state=None):# 裁剪关节位置
    start, _ = get_root_joint_dim(config.model)
    end = config.model.nq

    qpos = config.q.copy()
    qpos[start:end] = np.clip(
        qpos[start:end], 
        a_min=q_min[start:end], 
        a_max=q_max[start:end]
    )
    if root_state is not None:
        qpos[0:7] = root_state
    qpos.setflags(write=False)
    config.q = qpos
  

def blend_motion(array, skip):# ???不懂，但是是按照一个指定的skip步长进行分组，然后分组求平均值返回  ???
    array = array[:int(int(array.shape[0] / skip) * skip)]
    array = array.reshape((int(array.shape[0] / skip), skip, -1))
    return np.mean(array, axis=1)



if __name__ == "__main__":# 解析参数-初始化-指定任务-ik求解-存储
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", type=str, help="robot resource directory", required=True)
    parser.add_argument("-urdf", type=str, help="urdf directory and file name", required=True)
    parser.add_argument("-srdf", type=str, help="srdf directory and file name", required=True)
    parser.add_argument("-pkg", type=str, help="robot mesh directory", required=True)
    parser.add_argument("-yaml", type=str, required=True)
    
    parser.add_argument("-smpl", type=str, help="smpl model directory", required=True) # 2015 马普的一篇文章“SMPL: a skinned multi-person linear model”中构建的人体参数化三维模型
    parser.add_argument("-data", type=str, help="mocap data directory", required=True)
    parser.add_argument("-rate", type=int, help="target framerate", required=True)
    parser.add_argument("-output", type=str, help="output directory and file name", required=True)
    parser.add_argument("-headless", help="start without rendering", action="store_true")
    parser.add_argument("-print_joints", help="print joint indices and names", action="store_true")
    args = parser.parse_args()
    
    # Load SMPL model & robot URDF
    # SMPL是动捕数据库的；URDF是机器人的
    smpl_models = load_smpl_models(args.smpl)
    
    urdf_path = os.path.join(args.path, args.urdf)
    robot = pin.RobotWrapper.BuildFromURDF(filename=urdf_path, package_dirs=[args.pkg], # pinnochio实例的robot
                                           root_joint=pin.JointModelFreeFlyer())
    human = build_human_wrapper(args.smpl, root_joint=pin.JointModelFreeFlyer()) # pinnochio实例的human
    '''??? 为什么要有一个robot和一个human两个pinnochio实例呢??? 未知'''
    
    if not args.headless:
        robot_viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
        robot.setVisualizer(robot_viz, init=False);robot_viz.initViewer(open=True);robot_viz.loadViewerModel()
        
        human_viz = MeshcatVisualizer(human.model, human.collision_model, human.visual_model)
        robot.setVisualizer(human_viz, init=False);human_viz.initViewer(open=True);human_viz.loadViewerModel()
    
    config = pink.Configuration(robot.model, robot.data, robot.q0, collision_model=robot.collision_model)
    link_names = [frame.name for frame in robot.model.frames if frame.type == BODY]
    
    # Load YAML config
    # 这个是最开始的robot的config，例如h1_2.yml
    # 这里把YAML变成一个dictionary
    yaml_dict = yaml.load(open(args.yaml, "r", encoding="utf-8"), Loader=yaml.FullLoader)  
    
    
    
    # Setup T-Pose  
    # 应该是torso的意思
    t_pose = robot.q0.copy()
    if yaml_dict["TPoseJointPositions"]:
        for joint_name, degree in yaml_dict["TPoseJointPositions"].items():
            t_pose[robot.model.getJointId(joint_name)+5] = np.deg2rad(degree) # 转成弧度
    
    if "ClipMinJointPositions" in yaml_dict.keys():
        q_min_dict = yaml_dict["ClipMinJointPositions"]
    else: q_min_dict = {}
    if "ClipMaxJointPositions" in yaml_dict.keys():
        q_max_dict = yaml_dict["ClipMaxJointPositions"]
    else: q_max_dict = {}
    q_max, q_min = make_joint_config(args, robot, config, q_min_dict, q_max_dict) # 这个函数在前面有定义（最大最小角度限制）
    
    
    
    # Setup tasks
    # 读取YAML里FrameTask里面每一个目标关节的config
    # 然后把读到的config给task相关的字典赋值
    
    # 初始化task
    tasks, task_info, pos_indices, rpy_indices, extra_rot = {}, {}, {}, {}, {}
    for task_name, content in yaml_dict["FrameTasks"].items(): # item()就是取出这个值并一会返回；这样做比Index取值优势在于-精度高
        tasks[task_name] = FrameTask(
            content["link_name"],
            position_cost=content["cost"]["position"], 
            orientation_cost=content["cost"]["orientation"], 
            lm_damping=content["cost"]["damping"])
        pos_indices[task_name] = [get_smpl_id(j) for j in content["pos_smpl_joint"]]
        rpy_indices[task_name] = [get_smpl_id(j) for j in content["rpy_smpl_joint"]]
        task_info[task_name] = None
        
        extra_rpy = [np.deg2rad(i) for i in content["axes_rotation"]]
        extra_rot[task_name] = utils.rpy2mat(torch.tensor(extra_rpy, dtype=torch.float))
    
    
    
    # 初始化relative task
    bindings, hackings, targets = yaml_dict["SkeletonBinding"], yaml_dict["SkeletonHacking"], yaml_dict["TargetHacking"]
    for task_name, content in yaml_dict["RelativeFrameTasks"].items():
        tasks[task_name] = RelativeFrameTask(
            content["link_name"], bindings[yaml_dict["Root"]],
            position_cost=content["cost"]["position"], 
            orientation_cost=content["cost"]["orientation"], 
            lm_damping=content["cost"]["damping"])
        pos_indices[task_name] = [get_smpl_id(j) for j in content["pos_smpl_joint"]]
        rpy_indices[task_name] = [get_smpl_id(j) for j in content["rpy_smpl_joint"]]
        task_info[task_name] = content["root_name"]
        
        extra_rpy = [np.deg2rad(i) for i in content["axes_rotation"]]
        extra_rot[task_name] = utils.rpy2mat(torch.tensor(extra_rpy, dtype=torch.float))
        
        
        
    # 初始化CoMTask
    com_task = ComTask(cost=yaml_dict["CoMTask"]["cost"], 
        lm_damping=yaml_dict["CoMTask"]["damping"], gain=yaml_dict["CoMTask"]["gain"])
    acceleration_task = LowAccelerationTask(cost=yaml_dict["AccelerationTask"]["cost"])
          
    # 可视化
    if not args.headless:
        poses = {f"{k}_pose": transform_wrt_world(config, tasks[k].frame) for k in tasks.keys()}
        meshcat_shapes.frame(robot_viz.viewer["com_pos"])
        for key in poses.keys():
            meshcat_shapes.frame(robot_viz.viewer[key])
            robot_viz.viewer[key].set_transform(poses[key].np)
            
            
            
            
            
    # Initialize QP-solver       
    solver = qpsolvers.available_solvers[0]  # https://github.com/qpsolvers/qpsolvers 二次规划求解器，有约束下，最小化一个二次型函数quadratic programs
    if "proxqp" in qpsolvers.available_solvers: solver = "proxqp"
    print(f"Using {solver} QP solver.")      # ProxQP 引入所谓的 proximity operator（近似算子） 来简化优化问题的求解。

    
    smpl_axes_rot = utils.rpy2mat(torch.tensor([0., -np.pi/2, -np.pi/2], dtype=torch.float))
    

    for mocap_name, mocap_data in tqdm(pickle.load(open(args.data,"rb")).items()):  # mocap_name是键；mocap——data是值；遍历1次，取1次键key和1次值value

        # Re-initialize config
        v_template, shapedirs, J_regressor, parents = smpl_models[mocap_data["gender"]]         # 都是 SMPL 文件的参数
                                                                                                #     v_template 表示人体模型的基准顶点位置;通常是一个形状为 (V, 3) 的 NumPy,V 是顶点的数量，每个顶点有三个坐标（x, y, z）。
                                                                                                #     shapedirs 调整形状参数，可以改变人体的胖瘦、高矮等特征
                                                                                                #     J_regressor 是一个权重矩阵，用于计算每个关节的位置
                                                                                                #     parents     定义骨骼的层级关系
        # Load mocap trajectory
        skip = max(1, int(mocap_data["mocap_framerate"] / args.rate))                  #帧率/采样率 = 间隔skip
        betas = torch.tensor(mocap_data["betas"][np.newaxis, 0:10], dtype=torch.float) #形状参数betas(1维)转tensor,并新增一个维度
        full_pose = torch.tensor(blend_motion(mocap_data["poses"][:, 0:66], skip), dtype=torch.float)*0   # 局部debug # pose转tensor
        trans = torch.tensor(blend_motion(mocap_data["trans"], skip), dtype=torch.float)        # trans转tensor
                
                
                
        # SkeletonBinding Human and Humanoid
        v_shaped = v_template + torch.einsum("bl,mkl->bmk", [betas, shapedirs])
        J, ratio = binding.J_calibration(torch.einsum("bik,ji->bjk", [v_shaped, J_regressor]), 
                                         robot, t_pose, parents, bindings, hackings) # 这个函数返回的是joint和ratio
                                                                                     # 可以理解成 做对齐
        
        trans[:, 2] *= ratio;trans[:, 0:2] *= min(0.8, ratio)
        trans[:, 2] -= torch.min(v_shaped[..., 2], dim=1).values
        trans[:, 0:2] -= trans[0:1, 0:2].clone()
        
        # Obtain SMPL joints (without fingers and face)
        rot_mats = utils.batch_rodrigues(full_pose.reshape(-1, 3)).reshape(trans.shape[0], -1, 3, 3)
        expand_J = J.expand(trans.shape[0], *J.shape[1:])
        smpl_transforms = utils.batch_rigid_transform(rot_mats, expand_J, parents)[0]
        smpl_transforms[..., :3, :3] = smpl_transforms[..., :3, :3] @ smpl_axes_rot     # @表示矩阵乘法   
        smpl_transforms[..., :3, 3] += trans[:, None] - smpl_transforms[:, 0:1, :3, 3].clone() # .clone() 确保
                                                                                # 从 smpl_transforms 提取的部分不被修改       
        smpl_rot_mats = smpl_transforms[..., :3, :3].clone()
        smpl_parent_inv_rot_mats = torch.inverse(smpl_rot_mats[:, parents[1:]])
        smpl_rot_mats[:, 1:] = smpl_parent_inv_rot_mats @ smpl_rot_mats[:, 1:]
        human_joint_order = [smpl_body_names.index(n) for n in human_ordered_joint_names]
        human_rot_mats = smpl_rot_mats[:, human_joint_order]

        # Initialize result buffer
        results = {"base_orientation": [], "base_position": [], "joint_position": []}
        link_location = {k: [] for k in link_names if "keyframe_" in k}
        link_orientation = {k: [] for k in link_names if "keyframe_" in k}
        
        
        
        
        
        # Initiatlize height
        rate = RateLimiter(frequency=args.rate)
        for t in range(smpl_transforms.shape[0]): # 对每一个smpl的transform开始操作
            root_name = yaml_dict["Root"]
            root_id = get_smpl_id(root_name)
            root_transform = smpl_transforms[t, root_id]
                
            # Setup IK targets
            for key in tasks.keys():
                pos_ids, rpy_ids = pos_indices[key], rpy_indices[key]
                joint_transform = smpl_transforms[t, pos_ids[0]].clone()
                                
                if len(pos_ids) == 2:   # 用意是什么？
                    second_joint_pos = smpl_transforms[t, pos_ids[1], :3, 3]
                    joint_transform[:3, 3] += second_joint_pos  # joint transform是ik的target
                    joint_transform[:3, 3] *= 0.5
                elif len(pos_ids) > 2:
                    raise NotImplementedError
                
                # Caculate rotation matrix
                if smpl_body_names[rpy_ids[0]] == "head":
                    joint_transform[:3, :3] = extra_rot[key]
                else:
                    joint_transform[:3, :3] @= extra_rot[key]
                
                if task_info[key] is None:
                    frame_target = tensors2se3(joint_transform[:3, :3], joint_transform[:3, 3])
                    tasks[key].set_target(frame_target)
                    if not args.headless:
                        robot_viz.viewer[f"{key}_pose"].set_transform(frame_target.np) # f-string 格式化字符串，将变量 key 的值与 _pose 拼接成一个新的字符串
                else:
                    smpl_rel_root_name = task_info[key]
                    smpl_root_id = get_smpl_id(smpl_rel_root_name)
                    rel_root_name = bindings[smpl_rel_root_name]
                    robot_root_name = bindings[root_name] # 一个smpl的root定义，一个robot的root定义
                    
                    joint_transform_wrt_root = torch.inverse(root_transform) @ joint_transform
                    robot_rel_root_transform = torch.tensor(transform_wrt_world(config, rel_root_name).np, dtype=torch.float)
                    rel_root_transform_wrt_root = torch.inverse(root_transform) @ smpl_transforms[t, smpl_root_id] # smpl的
                    
                    robot_root_transform = torch.tensor(transform_wrt_world(config, robot_root_name).np, dtype=torch.float)
                    robot_rel_root_transform = torch.tensor(transform_wrt_world(config, rel_root_name).np, dtype=torch.float)
                    robot_rel_root_transform_wrt_root = torch.inverse(robot_root_transform) @ robot_rel_root_transform # robot的
                    
                    frame_target_rot = joint_transform_wrt_root[:3, :3] # $$$ 旋转 $$$
                    frame_target_pos = robot_rel_root_transform_wrt_root[:3, 3] + \
                        joint_transform_wrt_root[:3, 3] - rel_root_transform_wrt_root[:3, 3] # $$$ 位置 $$$ 不太懂
                        
                    # 可视化ik目标
                    if tasks[key].frame in bindings.values():
                        target_key = list(bindings.keys())[list(bindings.values()).index(tasks[key].frame)]
                        if target_key in targets.keys():                        
                            frame_target_pos += torch.tensor(targets[target_key], dtype=torch.float)
                        
                    frame_target = tensors2se3(frame_target_rot, frame_target_pos)
                    tasks[key].set_target(frame_target)
                    if not args.headless:
                        viz_frame_target = transform_wrt_world(config, robot_root_name) * frame_target
                        robot_viz.viewer[f"{key}_pose"].set_transform(viz_frame_target.np)
                    
            # Setup CoM task
            human_q, human_v, human_mat = human.q0.copy(), human.v0.copy(), human_rot_mats[t]
            human_q[0:7] = transfrom2q(smpl_transforms[t, 0])
            if t > 0:
                human_v[0:6] = transfrom2v(smpl_transforms[t, 0], smpl_transforms[t-1, 0], rate.period)
            for i in range(len(human_mat)):
                q_start, q_end = i * 3 + 7, (i + 1) * 3 + 7
                human_q[q_start:q_end] = utils.mat2rpy(human_mat[i]).numpy()
            com_target = pin.centerOfMass(human.model, human.data, human_q, human_v)
            com_task.set_target(com_target)
            if not args.headless:
                robot_com = pin.centerOfMass(robot.model, robot.data, config.q)
                viz_com_target = pin.SE3(root_transform[:3, :3].numpy(), robot_com)
                robot_viz.viewer["com_pos"].set_transform(viz_com_target.np)
            
            # Compute velocity and integrate it into next config 计算速度；然后积分integrate
            if t > 0:
                task_list = list(tasks.values()) + [acceleration_task, com_task]
                velocity = solve_ik(config, task_list, rate.period, solver=solver, safety_break=False) # 一个pink接口就解决了vel的求解
                config.integrate_inplace(velocity, rate.period)
                acceleration_task.set_last_integration(config, velocity, rate.period)
                qpos_clamp(config, q_max, q_min, root_state=None)
                
                tensor_q = torch.tensor(config.q, dtype=torch.float)
                tensor_v = torch.tensor(velocity, dtype=torch.float)
                results["base_orientation"].append(utils.quat2rpy(tensor_q[3:7]))
                results["base_position"].append(tensor_q[0:3])
                results["joint_position"].append(tensor_q[7:])
                                
                # Obtain link infomation & fix height
                min_link_height = 100
                for name in link_location.keys():
                    link_transform = torch.tensor(transform_wrt_world(config, name).np, dtype=torch.float)
                    body_rpy = utils.mat2rpy(link_transform[:3, :3]);body_pos = link_transform[:3, 3]
                    link_orientation[name] += [body_rpy];link_location[name] += [body_pos]
                    min_link_height = min(min_link_height, body_pos[2].item())
                    
                if min_link_height < MIN_HEIGHT_THRESHOLD: # 如果小于最小值
                    curr_q_pos = deepcopy(config.q)
                    curr_q_pos[2] += abs(min_link_height)
                    config.update(curr_q_pos)
                    
                    results["base_position"][-1][2] += abs(min_link_height) 
                    for name in link_location.keys():
                        link_location[name][-1][2] += abs(min_link_height)
                    
                if not args.headless:
                    robot_viz.display(config.q)
                    human_viz.display(human_q)
                    rate.sleep()
            else:
                task_list = list(tasks.values()) + [com_task]
                last_q_pos = deepcopy(robot.q0)
                config.update(last_q_pos)
                
                timestep = 0
                while True:
                    velocity = solve_ik(config, task_list, rate.period, solver=solver, safety_break=False)
                    config.integrate_inplace(velocity, rate.period)
                    root_state = transfrom2q(smpl_transforms[t, 0])
                    qpos_clamp(config, q_max, q_min, root_state=root_state)
                    
                    q_cos_err = np.abs(config.q - last_q_pos) < np.pi / 180.0
                    # more than 2 second
                    if q_cos_err.all() or timestep > 2 * args.rate:break
                    last_q_pos = config.q.copy();timestep += 1
                    
                    # fix height
                    min_link_height = 1e5
                    for name in link_location.keys():
                        link_transform = transform_wrt_world(config, name).np
                        min_link_height = min(min_link_height, link_transform[:3, 3][2])
                        
                    if min_link_height < MIN_HEIGHT_THRESHOLD:
                        curr_q_pos = deepcopy(config.q)
                        curr_q_pos[2] += abs(min_link_height)
                        config.update(curr_q_pos)
                        
                acceleration_task.set_last_integration(config, velocity, rate.period)
        
        
        
        
        
        # Stack results into a single numpy array
        final_results = {"framerate": args.rate} # rate是帧率，在传参数的时候传进去的
        # 储存的结果有以下三个：1、result(不知道具体是什么result) 2、link_orientation 3、link_location
        # 这些结果都可以在中间ik求解那里看得到
        final_results.update({k: torch.stack(v, dim=0)[1:].numpy() for k, v in results.items()})
        link_orientation = {k: torch.stack(v, dim=0)[1:].numpy() for k, v in link_orientation.items()}
        link_location = {k: torch.stack(v, dim=0)[1:].numpy() for k, v in link_location.items()}
        final_results["link_orientation"], final_results["link_position"] = link_orientation, link_location
        pickle.dump(final_results, open(f"{args.output}/{mocap_name[2:]}.npz", "wb")) 
        # 保存成一个文件，名字xxx.npz    
