import pickle
import argparse
import random
import os
import os.path as osp
from tqdm import tqdm


amass_nba_data = [
    "0-CMU_06_06_03_poses",
    "0-CMU_06_06_04_poses",
    "0-CMU_06_06_05_poses",
    "0-CMU_06_06_06_poses",
    "0-CMU_06_06_07_poses",
    "0-CMU_06_06_08_poses",
    "0-CMU_06_06_09_poses",
    "0-CMU_06_06_10_poses",
    "0-CMU_06_06_11_poses",
    "0-CMU_06_06_12_poses",
    "0-CMU_06_06_13_poses",
    "0-CMU_06_06_14_poses",
    "0-CMU_06_06_15_poses",
] # prefix "0-"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="amass_data/amass_db_smplh.pt")
    parser.add_argument("--out", type=str, default="smpl_data")
    args = parser.parse_args()
    
    amass_db = pickle.load(open(args.path,'rb')) # 使用 pickle.load 方法从文件中加载序列化的对象 amass_db
    
    random.shuffle(amass_nba_data) #  random.shuffle 方法随机打乱 amass_nba_data 列表。

    motion_types = {"nba": amass_nba_data} # 定义一个字典 motion_types，将运动类型（如 "nba"）与其对应的运动数据名称关联起来
    
    for motion_type, mocap_list in motion_types.items(): # 遍历 motion_types 字典中的每个键值对 (motion_type, mocap_list)。
        # 字典的 items() 方法返回一个包含字典所有键值对的视图对象。每次迭代时，返回一个键值对 (key, value)。
        '''
        motion_type : 这是字典的键，表示运动类型（如 "nba"）。
        mocap_list : 这是字典的值，表示与运动类型相关联的运动数据名称列表。
        
        假设 motion_types 的内容如下
            motion_types = {
                "nba": ["0-CMU_06_06_03_poses", "0-CMU_06_06_04_poses"],
                "football": ["0-CMU_06_06_05_poses", "0-CMU_06_06_06_poses"]
            }
            
        执行以下代码,输出结果将是：
            Motion Type: nba
            Mocap List: ['0-CMU_06_06_03_poses', '0-CMU_06_06_04_poses']
            Motion Type: football
            Mocap List: ['0-CMU_06_06_05_poses', '0-CMU_06_06_06_poses']
        
        '''
        dataset = {}
        
        for mocap_name in tqdm(mocap_list):
            start_name = mocap_name.split("-")[1]
            found = mocap_name in amass_db.keys()
            if not found:
                print(f"Not found!! {start_name}")
                continue
            dataset[mocap_name] = amass_db[mocap_name] # 将当前 mocap_name 对应的数据添加到 dataset 中。
        
        if not osp.exists(args.out): os.mkdir(args.out) # 检查输出目录是否存在，如果不存在则创建。
        pickle.dump(dataset, open(f"{args.out}/amass_{motion_type}_train.pkl", 'wb')) # 分类后的数据保存为单独的文件，文件名格式为 amass_{motion_type}_train.pkl。
