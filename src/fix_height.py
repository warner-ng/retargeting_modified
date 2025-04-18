import warnings
warnings.filterwarnings("ignore")

import os
import sys
import pickle
import argparse
import numpy as np
import collections
import pybullet as p
import pybullet_data

from tqdm import tqdm
from contextlib import contextmanager

NUM_SAMPLE = 3
NUM_SECOND = 1
INIT_HEIGHT = 1e4


def load_dataset(folder, suffix=".npz"):
    filenames = [name for name in os.listdir(folder) if name[-len(suffix):] == suffix]
    datatset, data_path = {}, {}
    for filename in tqdm(filenames):
        try:
            file_path = os.path.join(folder, filename)
            data = pickle.load(open(file_path, 'rb'))
            datatset[filename[:-len(suffix)]] = data
            data_path[filename[:-len(suffix)]] = file_path
        except:
            print(f"{filename} load failed!!!")
            continue
    return datatset, data_path


@contextmanager
def suppress_stdout():
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()
        os.dup2(to.fileno(), fd)
        sys.stdout = os.fdopen(fd, "w")

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(os.devnull, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield
        finally:
            _redirect_stdout(to=old_stdout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-robot", type=str, help="robot directory", default="robot/")
    parser.add_argument("-urdf", type=str, help="urdf path", required=True)
    parser.add_argument("-mapping", type=str, help="joint mapping file path", required=True)
    parser.add_argument("-folder", type=str, help="dataset folder", required=True)
    args = parser.parse_args()

    with suppress_stdout():
        p.connect(p.DIRECT)
        p.setGravity(0, 0, -10)
        p.setAdditionalSearchPath(args.robot)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        
        urdf_file = os.path.join(args.robot, args.urdf)
        init_state = ([0, 0, 1.3], p.getQuaternionFromEuler([0, 0, 0]))
        robot = p.loadURDF(urdf_file, *init_state)

    lines = [line[:-1].split(" ") for line in open(args.mapping).readlines()]
    mapping = {k: int(v) for v, k in lines}
    p_mapping = {}
    for id in range(p.getNumJoints(robot)):
        j_info = p.getJointInfo(robot, id)
        if j_info[2] == p.JOINT_REVOLUTE:
            p_mapping[str(j_info[1])[2:-1]] = j_info[0]

    datasets, data_paths = load_dataset(args.folder)

    for key, data in tqdm(datasets.items()):
        queue = collections.deque(maxlen=NUM_SAMPLE)
  
        num_frame = min(data["base_orientation"].shape[0], int(NUM_SECOND * data["framerate"]))
        link_heights = np.stack(list(data["link_position"].values()), axis=1)
        min_index = np.argmin(np.min(link_heights[:num_frame, :, 2], axis=1), axis=0)
        base_quat = p.getQuaternionFromEuler(data["base_orientation"][min_index])
        p.resetBasePositionAndOrientation(robot, [0, 0, 1.3], base_quat)
        
        timestep = 0
        last_height = INIT_HEIGHT
        while True:
            height = p.getBasePositionAndOrientation(robot)[0][2]
            p.resetBasePositionAndOrientation(robot, [0, 0, height], base_quat)
            for j_name, j_id in p_mapping.items():
                j_pos, id = data["joint_position"], mapping[j_name]
                p.resetJointState(robot, j_id, j_pos[min_index, id])
                
            p.stepSimulation()
            queue.append(height)
            
            if timestep > int(5e3):
                curr_height = data["base_position"][min_index, 2]
                mean_height = np.mean(np.array(list(queue)))
                for link_name in data["link_position"].keys():
                    data["link_position"][link_name][:, 2] -= curr_height
                    data["link_position"][link_name][:, 2] += mean_height
                     
                data["base_position"][:, 2] -= curr_height
                data["base_position"][:, 2] += (mean_height + 0.05)
                pickle.dump(data, open(data_paths[key], 'wb'))
                break
                
            last_height = height
            timestep += 1