# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

# 该脚本读取并处理 AMASS 数据集，区分干净数据与完整数据，剔除不良片段，
# 保存为两个序列化文件 (amass_clean_db_smplh.pt 和 amass_db_smplh.pt)，并输出数据统计信息及最低帧率

'''read_single_sequence 处理所有动作数据，而 read_clean_single_sequence 会根据遮挡信息剔除有问题的数据片段，确保数据更干净。'''

import os
import os.path as osp

import math
import joblib
import pickle
import argparse
import numpy as np
import os.path as osp
from tqdm import tqdm

min_framerate = 1e7
all_sequences = [
    # "ACCAD",
    # "BMLmovi",
    # "BioMotionLab_NTroje",
    # "CMU",
    # "DFaust_67",
    # "EKUT",
    # "Eyes_Japan_Dataset",
    # "HumanEva",
    # "KIT",
    # "MPI_HDM05",
    # "MPI_Limits",
    # "MPI_mosh",
    "SFU",
    # "SSM_synced",
    "TCD_handMocap",
    "TotalCapture",
    "Transitions_mocap",
    "BMLhandball",
    "DanceDB",
]

def read_data(folder):
    occlusions = joblib.load(osp.join(folder, "occlusion.pkl"))    

    sub_folders = [
        osp.join(folder, sub_folder) 
        for sub_folder in os.listdir(folder)
        if sub_folder in all_sequences
    ]    
    sub_folders = [
        sub_folder for sub_folder in sub_folders 
        if osp.exists(sub_folder) and osp.isdir(sub_folder)
    ]
    
    clean_db, db = {}, {}
    for sub_folder in sub_folders:
        print(f"Reading {sub_folder} sequence...")
        clean_datas = read_clean_single_sequence(sub_folder, sub_folder.split('/')[-1], occlusions)
        datas = read_single_sequence(sub_folder, sub_folder.split('/')[-1])
        clean_db.update(clean_datas)
        db.update(datas)
    return clean_db, db


def read_single_sequence(folder, seq_name):
    global min_framerate
    subjects = [
        subject for subject in os.listdir(folder) # 获取文件夹内的子文件夹（主题），过滤出存在的目录
        if osp.exists(osp.join(folder, subject)) \
            and osp.isdir(osp.join(folder, subject))
    ]
    
    datas = {} 
    for subject in tqdm(subjects): # 遍历每个主题，显示进度条
        actions = [
            x for x in os.listdir(osp.join(folder, subject)) 
            if x.endswith(".npz") and osp.isdir(osp.join(folder, subject))
        ] # 获取主题内的 `.npz` 文件，过滤出存在的文件
        
        for action in actions:
            fname = osp.join(folder, subject, action)
            if fname.endswith("shape.npz"): continue # 遍历每个动作文件，跳过 `shape.npz` 文件
            data = dict(np.load(fname, allow_pickle=True))
            
            mocap_framerate = math.ceil(data["mocap_framerate"])
            skip = int(mocap_framerate / 60)
            min_framerate = min(min_framerate, mocap_framerate) # 计算动作捕捉帧率并向下取整，更新全局最低帧率
            
            # 其他信息：处理动作捕捉数据，包括调整 betas 形状、下采样 trans 和 poses，
            # 生成唯一视频名，初始化数据结构，设置性别等属性，最后存储处理后的数据
            if len(data["betas"].shape) == 2:
                data["betas"] = data["betas"][0]
            data["trans"] = data["trans"][::skip]
            data["poses"] = data["poses"][::skip]
            
            vid_name = f"0-{seq_name}_{subject}_{action[:-4]}"
                            
            total_data = {"mocap_framerate": 60.0}
            for key in ["betas", "gender", "poses", "trans"]:
                if key == "gender":
                    if "fe" in data[key]: total_data[key] = "female"
                    elif "ne" in data[key]: total_data[key] = "neutral"
                    else: total_data[key] = "male"
                else:
                    if key in ["poses", "trans"]: total_data[key] = data[key]
                    else: total_data[key] = data[key]
            datas[vid_name] = total_data
            
    return datas


def read_clean_single_sequence(folder, seq_name, occlusions):
    global min_framerate
    subjects = [
        subject for subject in os.listdir(folder) 
        if osp.exists(osp.join(folder, subject)) \
            and osp.isdir(osp.join(folder, subject))
    ]                                               # osp是os.path的缩写
    
    clean_datas = {}
    for subject in tqdm(subjects):
        actions = [
            x for x in os.listdir(osp.join(folder, subject)) 
            if x.endswith(".npz") and osp.isdir(osp.join(folder, subject))
        ]
        
        for action in actions:
            bound = 0                               # bound是mocap的时长
            fname = osp.join(folder, subject, action)
            if fname.endswith("shape.npz"): continue
            data = dict(np.load(fname, allow_pickle=True))
            
            mocap_framerate = math.ceil(data["mocap_framerate"])
            skip = int(mocap_framerate / 60)
            min_framerate = min(min_framerate, mocap_framerate)
            
            if len(data["betas"].shape) == 2:
                data["betas"] = data["betas"][0]
            data["trans"] = data["trans"][::skip] # 切片操作 [start:end:step] 从序列中提取子集
            data["poses"] = data["poses"][::skip]

            vid_name = f"0-{seq_name}_{subject}_{action[:-4]}"
            
            if vid_name in occlusions.keys(): # occlusions 是一个字典，存储了视频片段的相关信息，包括潜在的问题（如 "sitting" 或 "airborne"）
                                                # 及相关的索引信息 "idxes"。用于帮助判断和处理数据中的遮挡或其他异常情况。
                issue = occlusions[vid_name]["issue"]
                if (issue == "sitting" or issue == "airborne") and \
                    "idxes" in occlusions[vid_name]:
                    bound = occlusions[vid_name]["idxes"][0] * 2
                    # This bounded is calucaled assuming 60 FPS.....
                    if bound < 20:
                        print("bound too small", vid_name, bound)
                        continue
                else:
                    print("issue irrecoverable", vid_name, issue)
                    continue
            
            if "0-KIT_442_PizzaDelivery02_poses" == vid_name:
                bound = -4
                
            if bound == 0:
                bound = data["trans"].shape[0]  # .shape[0] 获取数组或矩阵的第一维大小，即行数
                
            if data["trans"].shape[0] < 20:
                continue
                                        
            clean_data = {"mocap_framerate": 60.0}
            for key in ["betas", "gender", "poses", "trans"]:                
                if key == "gender":
                    if "fe" in data[key]: clean_data[key] = "female"
                    elif "ne" in data[key]: clean_data[key] = "neutral"
                    else: clean_data[key] = "male"
                else:
                    if key in ["poses", "trans"] and bound != 0:
                        clean_data[key] = data[key][:bound]
                    else: clean_data[key] = data[key]
                   
            clean_datas[vid_name] = clean_data
            
    return clean_datas


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="clean_dataset directory", default="data")
    parser.add_argument("--out", type=str, help="output directory", default="amass_data")
    args = parser.parse_args()
    
    if not osp.exists(args.out): os.mkdir(args.out)
    clean_db_file = osp.join(args.out, "amass_clean_db_smplh.pt")
    db_file = osp.join(args.out, "amass_db_smplh.pt")
    print(f"Saving AMASS clean dataset to {clean_db_file}.")
    print(f"Saving AMASS total dataset to {db_file}.")

    clean_db, db = read_data(args.dir)
    clean_db_len, db_len = len(clean_db), len(db)
    
    pickle.dump(clean_db, open(clean_db_file, 'wb'));del clean_db
    pickle.dump(db, open(db_file, 'wb'));del db
    print(f"Total number of clean dataset is {clean_db_len}.")
    print(f"Total number of total dataset is {db_len}.")
    print(f"The minimum framerate of AMASS is {min_framerate}")