# 根据 URDF 文件生成 SRDF 文件 sementic robot descriptrion file


import os
import argparse
from urdf_parser_py.urdf import URDF

# # 加载URDF文件
# robot_name = "g1"
# urdf_file = f"robot/G1/urdf/{robot_name}.urdf"
# robot = URDF.from_xml_file(urdf_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-name", type=str, help="name of the robot", required=True)
    parser.add_argument("-path", type=str, help="urdf directory", required=True)
    args = parser.parse_args()
    
    urdf_file = os.path.join(args.path, f"{args.name}.urdf")
    robot = URDF.from_xml_file(urdf_file)
    
    def get_link(link_name):
        for link in robot.links:
            if link.name == link_name:
                return link
        
    def get_parent(link_name):
        for joint in robot.joints:
            if joint.child == link_name:
                for link in robot.links:
                    if link.name == joint.parent:
                        return link
    
    srdf_file = os.path.join(args.path, f"{args.name}.srdf")
    with open(srdf_file, "w") as file:
        file.write(f"""
<?xml version="1.0" encoding="UTF-8"?>
<!--This does not replace URDF, and is not an extension of URDF. 
    This is a format for representing semantic information about the robot structure.
    A URDF file must exist for this robot as well, where the joints and the links that are referenced are defined. -->
<robot name="{args.name}">
    <!--DISABLE COLLISIONS: By default it is assumed that any link of the robot could potentially come into collision with any other link in the robot. 
    This tag disables collision checking between a specified pair of links. -->""")

        collision_list = []
        for joint in robot.joints:
            parent_link = get_link(joint.parent)
            child_link = get_link(joint.child)
            if len(parent_link.collisions) > 0 and len(child_link.collisions) > 0:
                collision_list.append([parent_link.name, child_link.name])
                file.write(f"""
    <disable_collisions link1="{parent_link.name}" link2="{child_link.name}" reason="Adjacent"/>""")
        file.write("\n")
        
        for link in robot.links:
            if "keyframe_" in link.name and ("ankle" in link.name or "wrist" in link.name):
                end_effortor_link = get_parent(link.name)
                for link in robot.links:
                    if len(end_effortor_link.collisions) > 0 and len(link.collisions) > 0:
                        no_repeat = True
                        for collision in collision_list:
                            if not no_repeat: break
                            if set(collision) == set([end_effortor_link.name, link.name]):
                                no_repeat = False
                        
                        if no_repeat:
                            collision_list.append([end_effortor_link.name, link.name])
                            file.write(f"""
    <disable_collisions link1="{end_effortor_link.name}" link2="{link.name}" reason="Never"/>""")

        file.write("\n</robot>")
    
    