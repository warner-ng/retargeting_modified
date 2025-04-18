import os
import pickle
import argparse
from tqdm import tqdm # 用于显示进度条
 

amass_splits = {
    'valid': ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
    'test': ['Transitions_mocap', 'SSM_synced'],
    'train': [
        "CMU", 
        "MPI_Limits", 
        "TotalCapture", 
        "Eyes_Japan_Dataset", 
        "KIT", "EKUT", 
        "TCD_handMocap", 
        "BMLhandball", 
        "DanceDB", 
        "ACCAD", 
        "            ",
        "BioMotionLab_NTroje",
        "Eyes_Japan_Dataset", 
        "DFaust_67",
        ]  # Adding ACCAD
}

amass_split_dict = {}
for k, v in amass_splits.items():
    for d in v:
        amass_split_dict[d] = k

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="amass_data/amass_clean_db_smplh.pt")
    parser.add_argument("--out", type=str, default="smpl_data")
    args = parser.parse_args()
    
    amass_db = pickle.load(open(args.path,'rb'))
    train_data, test_data, valid_data = {}, {}, {}    
    for mocap_name, mocap in tqdm(amass_db.items()): # 遍历 amass_db 字典中的每个键值对 (mocap_name, mocap)
        start_name = mocap_name.split("-")[1]
        
        for dataset_key in amass_split_dict.keys():
            if start_name.startswith(dataset_key):
                split = amass_split_dict[dataset_key]
                
                if split == "test":
                    test_data[mocap_name] = amass_db[mocap_name]
                elif split == "valid":
                    valid_data[mocap_name] = amass_db[mocap_name]
                elif split == "train":
                    train_data[mocap_name] = amass_db[mocap_name]
                    
    print("Train: ", len(train_data), "\nValid: ", len(valid_data), "\nTest: ", len(test_data)) # 打印训练集、验证集和测试集的大小
    if not os.path.exists(args.out): os.mkdir(args.out) # 检查输出目录是否存在，如果不存在则创建。
    pickle.dump(train_data, open(f"{args.out}/amass_full_train.pkl", 'wb')) # 分类后的数据保存为单独的文件
    pickle.dump(valid_data, open(f"{args.out}/amass_full_valid.pkl", 'wb'))
    pickle.dump(test_data, open(f"{args.out}/amass_full_test.pkl", 'wb'))