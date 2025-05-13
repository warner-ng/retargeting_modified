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

get_smpl_id = lambda n: smpl_body_names.index(n)                                            # 根据SMPL模型的关节名称返回其索引
transform_wrt_world = lambda config, body: config.get_transform_frame_to_world(body).copy() # 获取某个物体相对于世界坐标系的变换矩阵。
tensors2se3 = lambda mat, vec: pin.SE3(mat.numpy(), vec.numpy())                            # 将旋转矩阵和平移向量转换为SE3变换
get_key_from_value = lambda d, v: list(d.keys())[list(d.values()).index(v)]                 # 根据字典值查找对应的键
transfrom2q = lambda x: torch.cat([x[:3, 3], utils.mat2quat(x[:3, :3])], dim=0).numpy()     # 将变换矩阵转换为四元数表示
transfrom2v = lambda x, y, t: torch.cat([(x[:3, 3] - y[:3, 3]) / t,                         # 计算两个变换之间的速度
    (utils.mat2rpy(x[:3, :3]) - utils.mat2rpy(y[:3, :3])) / t], dim=0).numpy()


def build_human_wrapper(path, filename="human.xml", root_joint=None):
    model, collision_model, visual_model = pin.shortcuts.buildModelsFromMJCF(
        os.path.join(path, filename), root_joint=root_joint)
    human = pin.RobotWrapper(model=model, collision_model=collision_model, visual_model=visual_model)
    return human

def get_root_joint_dim(model):
    if model.existJointName("root_joint"):
        root_joint_id = model.getJointId("root_joint")
        root_joint = model.joints[root_joint_id]
        return root_joint.nq, root_joint.nv
    return 0, 0

def load_smpl_models(folder):
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
        smpl_models[gender] = (v_template, shapedirs, J_regressor[:22], parents[:22])
    return smpl_models

def make_joint_config(args, robot, config, q_min_dict, q_max_dict):
    q_max = config.model.upperPositionLimit.copy()
    q_min = config.model.lowerPositionLimit.copy()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
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
    q_scale = (q_max[start:end] - q_min[start:end]) * 0.95
    q_max[start:end] = q_mean + q_scale * 0.5
    q_min[start:end] = q_mean - q_scale * 0.5
    for joint_name in robot.model.names:
        joint_id = robot.model.getJointId(joint_name) - 2 + start
        if joint_name in q_min_dict.keys():
            q_min[joint_id] = max(q_min[joint_id], np.deg2rad(q_min_dict[joint_name]))
        if joint_name in q_max_dict.keys():
            q_max[joint_id] = min(q_max[joint_id], np.deg2rad(q_max_dict[joint_name]))
    return q_max, q_min

def qpos_clamp(config, q_max, q_min, root_state=None):
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

def blend_motion(array, skip):
    array = array[:int(int(array.shape[0] / skip) * skip)]
    array = array.reshape((int(array.shape[0] / skip), skip, -1))
    return np.mean(array, axis=1)

def add_object(given_robot, frame_in_robot, object_urdf, object_package_dirs):
    # 从 URDF 加载新物体
    new_object = pin.RobotWrapper.BuildFromURDF(
        filename=object_urdf,
        package_dirs=[object_package_dirs],
        root_joint=pin.JointModelFreeFlyer()
    )

    # 获取位姿并计算相对变换矩阵 aMb
    T_B_world = new_object.data.oMf[0]
    T_A_world = given_robot.data.oMi[0]
    aMb = T_B_world * T_A_world.inverse()

    print("aMb:", aMb)
    print("Number of joints in new_object:", new_object.model.nq)

    # 重命名新物体的关节和帧名以避免冲突
    for i in range(new_object.model.njoints):
        new_object.model.names[i] = f"obj_{new_object.model.names[i]}"
    for i in range(len(new_object.model.frames)):
        new_object.model.frames[i].name = f"obj_{new_object.model.frames[i].name}"

    old_nq = given_robot.model.nq

    # 合并dynamic model（机器人 + 新物体）
    given_robot.model = pin.appendModel(
        modelA=given_robot.model,
        modelB=new_object.model,
        frame_in_modelA=frame_in_robot,
        aMb=aMb
    )

    # 手动将新物体的Geometry Model添加到主机器人模型中
    for geom in new_object.collision_model.geometryObjects:
        given_robot.collision_model.addGeometryObject(geom)
    for geom in new_object.visual_model.geometryObjects:
        given_robot.visual_model.addGeometryObject(geom)

    # 打印合并后的Geometry Model内容
    print("Collision Model Geometry Objects (after adding):")
    for i, geom in enumerate(given_robot.collision_model.geometryObjects):
        print(f"  {i}: {geom.name}, parent_joint={geom.parentJoint}")

    print("Visual Model Geometry Objects (after adding):")
    for i, geom in enumerate(given_robot.visual_model.geometryObjects):
        print(f"  {i}: {geom.name}, parent_joint={geom.parentJoint}")

    # 修正 q0 的长度
    print("given_robot.q0 is ",given_robot.q0)
    print("given_robot.model.nq is ",given_robot.model.nq)
    if len(given_robot.q0) != given_robot.model.nq:
        given_robot.q0 = np.pad(given_robot.q0, (0, given_robot.model.nq - len(given_robot.q0)), 'constant')
    print("now, given_robot.q0 is ",given_robot.q0)    
        
    robot_q0 = given_robot.q0[:old_nq]
    object_q0 = given_robot.q0[old_nq:]
    
    import ipdb
    ipdb.set_trace()  # 这里设置断点
    return given_robot, robot_q0, object_q0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", type=str, help="robot resource directory", required=True)
    parser.add_argument("-urdf", type=str, help="urdf directory and file name", required=True)
    parser.add_argument("-srdf", type=str, help="srdf directory and file name", required=True)
    parser.add_argument("-pkg", type=str, help="robot mesh directory", required=True)
    parser.add_argument("-yaml", type=str, required=True)
    parser.add_argument("-smpl", type=str, help="smpl model directory", required=True)
    parser.add_argument("-data", type=str, help="mocap data directory", required=True)
    parser.add_argument("-rate", type=int, help="target framerate", required=True)
    parser.add_argument("-output", type=str, help="output directory and file name", required=True)
    parser.add_argument("-headless", help="start without rendering", action="store_true")
    parser.add_argument("-print_joints", help="print joint indices and names", action="store_true")

     # 新增的对象相关参数
    parser.add_argument("-add_obj", help="whether to add object", action="store_true")
    parser.add_argument("-obj_name", type=str, help="object name")
    parser.add_argument("-obj_urdf", type=str, help="object urdf file path")
    parser.add_argument("-obj_pkg", type=str, help="object package directory")
    parser.add_argument("-frame_id", type=str, help="robot frame id")
    args = parser.parse_args()
    
    # Load SMPL model & robot URDF
    smpl_models = load_smpl_models(args.smpl)
    urdf_path = os.path.join(args.path, args.urdf)
    robot = pin.RobotWrapper.BuildFromURDF(filename=urdf_path, package_dirs=[args.pkg], root_joint=pin.JointModelFreeFlyer())
    human = build_human_wrapper(args.smpl, root_joint=pin.JointModelFreeFlyer())
    
    # 看看有什么frames
    print("numbers of frames is", robot.model.nframes)
    print("names of frames is", [frame.name for frame in robot.model.frames])
    
    # 固定到哪个frame上
    robot_frame_id = robot.model.getFrameId(args.frame_id)

    # Add object (e.g. ball)
    if args.add_obj:
        robot, robot_q0, object_q0 = add_object(robot, robot_frame_id, args.obj_urdf,args.obj_pkg)

    # 创建 robot 的可视化
    robot_viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
    robot_viz.initViewer(open=True)
    robot_viz.loadViewerModel()
    
    # 火柴人的可视化
    human_viz = MeshcatVisualizer(human.model, human.collision_model, human.visual_model)
    robot.setVisualizer(human_viz, init=False)
    human_viz.initViewer(open=True)
    human_viz.loadViewerModel()
    

    # Initialize configuration
    robot.data = pin.Data(robot.model)
    print("after data-update, data is ",robot.data)
    config = pink.Configuration(model = robot.model,
                                data = robot.data,
                                q = robot.q0,
                                collision_model=None,
                                collision_data=None)
    link_names = [frame.name for frame in robot.model.frames if frame.type == BODY]
    

    # 加载 YAML 配置
    yaml_dict = yaml.load(open(args.yaml, "r", encoding="utf-8"), Loader=yaml.FullLoader)  
    
       
    # Setup T-Pose  
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
    tasks, task_info, pos_indices, rpy_indices, extra_rot = {}, {}, {}, {}, {}
    for task_name, content in yaml_dict["FrameTasks"].items():
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
    
    # Initialize relative tasks
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
    
    # Initialize CoMTask
    com_task = ComTask(cost=yaml_dict["CoMTask"]["cost"], 
        lm_damping=yaml_dict["CoMTask"]["damping"], gain=yaml_dict["CoMTask"]["gain"])
    acceleration_task = LowAccelerationTask(cost=yaml_dict["AccelerationTask"]["cost"])
    
    # Visualize
    if not args.headless:
        poses = {f"{k}_pose": transform_wrt_world(config, tasks[k].frame) for k in tasks.keys()}
        meshcat_shapes.frame(robot_viz.viewer["com_pos"])
        for key in poses.keys():
            meshcat_shapes.frame(robot_viz.viewer[key])
            robot_viz.viewer[key].set_transform(poses[key].np)
    
    # Initialize QP-solver
    solver = qpsolvers.available_solvers[0]
    if "proxqp" in qpsolvers.available_solvers: solver = "proxqp"
    print(f"Using {solver} QP solver.")
    
    smpl_axes_rot = utils.rpy2mat(torch.tensor([0., -np.pi/2, -np.pi/2], dtype=torch.float))
    
    for mocap_name, mocap_data in tqdm(pickle.load(open(args.data,"rb")).items()):
        
        # Re-initialize config
        v_template, shapedirs, J_regressor, parents = smpl_models[mocap_data["gender"]]         
        # 都是 SMPL 文件的参数
        # v_template 表示人体模型的基准顶点位置;通常是一个形状为 (V, 3) 的 NumPy,V 是顶点的数量，每个顶点有三个坐标（x, y, z）。
        # shapedirs 调整形状参数，可以改变人体的胖瘦、高矮等特征
        # J_regressor 是一个权重矩阵，用于计算每个关节的位置
        # parents     定义骨骼的层级关系
        
        # Load mocap data
        smpl_models = load_smpl_models(args.smpl)
        skip = max(1, int(mocap_data["mocap_framerate"] / args.rate))
        betas = torch.tensor(mocap_data["betas"][np.newaxis, 0:10], dtype=torch.float)
        full_pose = torch.tensor(blend_motion(mocap_data["poses"][:, 0:66], skip), dtype=torch.float)
        trans = torch.tensor(blend_motion(mocap_data["trans"], skip), dtype=torch.float)
        
        # SkeletonBinding Human and Humanoid
        v_shaped = v_template + torch.einsum("bl,mkl->bmk", [betas, shapedirs])
        J, ratio = binding.J_calibration(torch.einsum("bik,ji->bjk", [v_shaped, J_regressor]), 
                                        robot, t_pose, parents, bindings, hackings)
        trans[:, 2] *= ratio;trans[:, 0:2] *= min(0.8, ratio)
        trans[:, 2] -= torch.min(v_shaped[..., 2], dim=1).values
        trans[:, 0:2] -= trans[0:1, 0:2].clone()
        
        # Obtain SMPL joints (without fingers and face)
        rot_mats = utils.batch_rodrigues(full_pose.reshape(-1, 3)).reshape(trans.shape[0], -1, 3, 3)
        expand_J = J.expand(trans.shape[0], *J.shape[1:])
        smpl_transforms = utils.batch_rigid_transform(rot_mats, expand_J, parents)[0]
        smpl_transforms[..., :3, :3] = smpl_transforms[..., :3, :3] @ smpl_axes_rot
        smpl_transforms[..., :3, 3] += trans[:, None] - smpl_transforms[:, 0:1, :3, 3].clone()
        
        smpl_rot_mats = smpl_transforms[..., :3, :3].clone()
        smpl_parent_inv_rot_mats = torch.inverse(smpl_rot_mats[:, parents[1:]])
        smpl_rot_mats[:, 1:] = smpl_parent_inv_rot_mats @ smpl_rot_mats[:, 1:]
        human_joint_order = [smpl_body_names.index(n) for n in human_ordered_joint_names]
        human_rot_mats = smpl_rot_mats[:, human_joint_order]

        
        # Initialize result buffer
        results = {"base_orientation": [], "base_position": [], "joint_position": []}
        link_location = {k: [] for k in link_names if "keyframe_" in k}
        link_orientation = {k: [] for k in link_names if "keyframe_" in k}
        
        # Initialize height
        rate = RateLimiter(frequency=args.rate)
        for t in range(smpl_transforms.shape[0]):
            print("robot_q0 is", robot_q0)
            print("object_q0 is ", object_q0)
            import ipdb; ipdb.set_trace()
            
            root_name = yaml_dict["Root"]
            root_id = get_smpl_id(root_name)
            root_transform = smpl_transforms[t, root_id]
            
            # Setup IK targets
            for key in tasks.keys():
                pos_ids, rpy_ids = pos_indices[key], rpy_indices[key]
                joint_transform = smpl_transforms[t, pos_ids[0]].clone()
                if len(pos_ids) == 2:
                    second_joint_pos = smpl_transforms[t, pos_ids[1], :3, 3]
                    joint_transform[:3, 3] += second_joint_pos
                    joint_transform[:3, 3] *= 0.5
                elif len(pos_ids) > 2:
                    raise NotImplementedError
                
                # Calculate rotation matrix
                if smpl_body_names[rpy_ids[0]] == "head":
                    joint_transform[:3, :3] = extra_rot[key]
                else:
                    joint_transform[:3, :3] @= extra_rot[key]
                
                if task_info[key] is None:
                    frame_target = tensors2se3(joint_transform[:3, :3], joint_transform[:3, 3])
                    tasks[key].set_target(frame_target)
                    if not args.headless:
                        robot_viz.viewer[f"{key}_pose"].set_transform(frame_target.np)
                else:
                    smpl_rel_root_name = task_info[key]
                    smpl_root_id = get_smpl_id(smpl_rel_root_name)
                    rel_root_name = bindings[smpl_rel_root_name]
                    robot_root_name = bindings[root_name]
                    joint_transform_wrt_root = torch.inverse(root_transform) @ joint_transform
                    robot_rel_root_transform = torch.tensor(transform_wrt_world(config, rel_root_name).np, dtype=torch.float)
                    rel_root_transform_wrt_root = torch.inverse(root_transform) @ smpl_transforms[t, smpl_root_id]
                    robot_root_transform = torch.tensor(transform_wrt_world(config, robot_root_name).np, dtype=torch.float)
                    robot_rel_root_transform_wrt_root = torch.inverse(robot_root_transform) @ robot_rel_root_transform
                    frame_target_rot = joint_transform_wrt_root[:3, :3]
                    frame_target_pos = robot_rel_root_transform_wrt_root[:3, 3] + \
                        joint_transform_wrt_root[:3, 3] - rel_root_transform_wrt_root[:3, 3]
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
            
            # Compute velocity and integrate it into next config
            if t > 0:
                task_list = list(tasks.values()) + [acceleration_task, com_task]
                velocity = solve_ik(config, task_list, rate.period, solver=solver, safety_break=False)
                config.integrate_inplace(velocity, rate.period)
                acceleration_task.set_last_integration(config, velocity, rate.period)
                qpos_clamp(config, q_max, q_min, root_state=None)
                tensor_q = torch.tensor(config.q, dtype=torch.float)
                tensor_v = torch.tensor(velocity, dtype=torch.float)
                results["base_orientation"].append(utils.quat2rpy(tensor_q[3:7]))
                results["base_position"].append(tensor_q[0:3])
                results["joint_position"].append(tensor_q[7:])
                
                # Obtain link information & fix height
                min_link_height = 100
                for name in link_location.keys():
                    link_transform = torch.tensor(transform_wrt_world(config, name).np, dtype=torch.float)
                    body_rpy = utils.mat2rpy(link_transform[:3, :3]);body_pos = link_transform[:3, 3]
                    link_orientation[name] += [body_rpy];link_location[name] += [body_pos]
                    min_link_height = min(min_link_height, body_pos[2].item())
                if min_link_height < MIN_HEIGHT_THRESHOLD:
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
                    if q_cos_err.all() or timestep > 2 * args.rate:
                        break
                    last_q_pos = config.q.copy();timestep += 1
                    
                    # Fix height
                    min_link_height = 1e5
                    for name in link_location.keys():
                        link_transform = transform_wrt_world(config, name).np
                        min_link_height = min(min_link_height, link_transform[:3, 3][2])
                    if min_link_height < MIN_HEIGHT_THRESHOLD:
                        curr_q_pos = deepcopy(config.q)
                        curr_q_pos[2] += abs(min_link_height)
                        config.update(curr_q_pos)
                    if not args.headless:
                        robot_viz.display(config.q)
                        human_viz.display(human_q)
                        rate.sleep()
        
        # Stack results into a single numpy array
        final_results = {"framerate": args.rate}
        final_results.update({k: torch.stack(v, dim=0)[1:].numpy() for k, v in results.items()})
        link_orientation = {k: torch.stack(v, dim=0)[1:].numpy() for k, v in link_orientation.items()}
        link_location = {k: torch.stack(v, dim=0)[1:].numpy() for k, v in link_location.items()}
        final_results["link_orientation"], final_results["link_position"] = link_orientation, link_location
        pickle.dump(final_results, open(f"{args.output}/{mocap_name[2:]}.npz", "wb"))