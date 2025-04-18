import os
import sys
import time
import pickle
import argparse
import pybullet_data
import pybullet as p

from tqdm import trange
from contextlib import contextmanager
from PyQt5.QtWidgets import QApplication, QMainWindow


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
            

class DropAPP(QMainWindow):
    def __init__(self, urdf, robot, mapping):
        super().__init__()
        self.title = 'DROP YOUR MOCAP FILE HERE!'
        self.setWindowTitle(self.title)
        self.resize(500, 300)
        self.setAcceptDrops(True)

        with suppress_stdout():
            p.connect(p.GUI)
            p.setGravity(0, 0, -9.81)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.loadURDF("plane.urdf")

            self.init_base_state = ([0, 0, 1.5], p.getQuaternionFromEuler([0, 0, 0]))
            p.setAdditionalSearchPath(robot)
            self.robot = p.loadURDF(os.path.join(robot, urdf), *self.init_base_state)
            print("wait for 10 seconds");time.sleep(3)
        
        self.markers = None
        self.p_mapping = {}
        for id in range(p.getNumJoints(self.robot)):
            j_info = p.getJointInfo(self.robot, id)
            if j_info[2] == p.JOINT_REVOLUTE: 
                self.p_mapping[str(j_info[1])[2:-1]] = j_info[0]

        lines = [line[:-1].split(" ") for line in open(mapping).readlines()]
        self.mapping = {k: int(v) for v, k in lines}

    def dragEnterEvent(self, event):
        event.accept() if event.mimeData().hasUrls() else event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        for file in files:
            try:
                data = pickle.load(open(file, 'rb'))
                
                if self.markers is None:
                    p.setAdditionalSearchPath("./visualize/")
                    self.markers = {n: p.loadURDF("marker.urdf", *self.init_base_state) 
                        for n in data["link_position"].keys()}
                                    
                for t in trange(data["base_position"].shape[0]):
                    self.set_robot_base_state(data, t)
                    self.set_robot_joint_state(data, t)
                    self.set_marker_position(data, t)
                    p.stepSimulation()
                    time.sleep(1 / data["framerate"])
                
                p.resetBasePositionAndOrientation(self.robot, *self.init_base_state)
                for name in self.markers.keys():
                    p.resetBasePositionAndOrientation(self.markers[name], *self.init_base_state)
                for j_id in self.p_mapping.values():
                    p.resetJointState(self.robot, j_id, 0.0)
                    
            except:
                continue
            
    def set_robot_joint_state(self, data, t):
        for j_name, j_id in self.p_mapping.items():
            j_pos, id = data["joint_position"], self.mapping[j_name]
            p.resetJointState(self.robot, j_id, j_pos[t, id])
    
    def set_robot_base_state(self, data, t):
        base_pos, base_rpy = data["base_position"][t], data["base_orientation"][t]
        base_quat = p.getQuaternionFromEuler(base_rpy)
        p.resetBasePositionAndOrientation(self.robot, base_pos, base_quat)
        
    def set_marker_position(self, data, t):
        link_position = data["link_position"]
        link_orientation = data["link_orientation"]
        
        for name in link_position.keys():
            marker_pos = link_position[name][t]
            marker_quat = p.getQuaternionFromEuler(link_orientation[name][t])
            p.resetBasePositionAndOrientation(self.markers[name], marker_pos, marker_quat)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-robot", type=str, help="robot directory", default="robot/")
    parser.add_argument("-urdf", type=str, help="urdf path", required=True)
    parser.add_argument("-mapping", type=str, help="joint mapping file path", required=True)
    args = parser.parse_args()

    with suppress_stdout():
        app = QApplication(sys.argv)
        window = DropAPP(args.urdf, args.robot, args.mapping)
        window.show();app.exec_()
        p.disconnect();sys.exit()

