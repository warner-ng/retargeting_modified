# import pinocchio as pin
# from pinocchio import GeometryObject, SE3


# # 创建 GeometryModel 并添加对象
# model = pin.GeometryModel()

# print(model.existGeometryName("warner"))

# new_model = pin.Model()
# print(new_model.nq)








# import pinocchio as pin
# import numpy as np

# # 定义一个惯性对象
# mass = 1.0
# lever = np.array([0.0, 0.0, 0.0])  # 质心相对于关节的位置
# inertia = np.diag([0.1, 0.1, 0.1])  # 惯性矩阵

# # 创建一个 Model 实例
# model1 = pin.Model()
# model2 = pin.Model()

# # 添加一些关节和身体以构建模型
# joint_model = pin.JointModelFreeFlyer()
# model1.addJoint(0, joint_model, pin.SE3.Identity(), "root_joint")

# # 创建 Inertia 对象
# body_inertia = pin.Inertia(mass, lever, inertia)
# model1.appendBodyToJoint(0, body_inertia, pin.SE3.Identity())

# # 创建一个 Data 实例
# data = model1.createData()

# # 检查数据与模型的一致性
# # is_consistent = model1.check(data)
# is_consistent = model2.check(data)

# if is_consistent:
#     print("数据与模型一致。")
# else:
#     print("数据与模型不一致。")






'''
下面介绍GeometryModel这个类
'''
import pinocchio as pin

geom_model = pin.GeometryModel()

import pinocchio as pin
import numpy as np



# 添加两个几何对象
sphere1 = pin.GeometryObject(
    name="sphere1",
    parent_joint=0,
    placement=pin.SE3.Identity(),
    collision_geometry=None,
    mesh_path="",
    mesh_scale=np.array([1.0, 1.0, 1.0]),
    override_material=False,
    mesh_color=np.array([1.0, 0.0, 0.0, 1.0]),
    mesh_texture_path="",
    mesh_material=pin.GeometryNoMaterial()
)

sphere2 = pin.GeometryObject(
    name="sphere2",
    parent_joint=0,
    placement=pin.SE3.Identity(),
    collision_geometry=None,
    mesh_path="",
    mesh_scale=np.array([1.0, 1.0, 1.0]),
    override_material=False,
    mesh_color=np.array([0.0, 1.0, 0.0, 1.0]),
    mesh_texture_path="",
    mesh_material=pin.GeometryNoMaterial()
)

# 将几何对象添加到 GeometryModel
geom_model.addGeometryObject(sphere1)
geom_model.addGeometryObject(sphere2)

print(sphere1)
print(sphere1.name)         # 几何对象名称
print(sphere1.placement)    # 位姿 (SE3)
print(sphere1.disableCollision) 
print(sphere1.meshColor) 

print(dir(sphere1))

# 假设我们有两个几何对象，索引分别是 0 和 1
collision_pair = pin.CollisionPair(0, 1)

# 添加这个碰撞对
geom_model.addCollisionPair(collision_pair)

if geom_model.existCollisionPair(collision_pair):
    print("该碰撞对已存在")
else:
    print("该碰撞对不存在")


index = geom_model.findCollisionPair(collision_pair)
print(f"碰撞对的索引是: {index}")

# 删除名为 "sphere" 的几何对象
geom_model.removeGeometryObject("sphere1")

# 创建 GeometryData
data = geom_model.createData()

geom_model.saveToBinary("geometry_model.bin")
loaded_geom_model = pin.GeometryModel()
loaded_geom_model.loadFromBinary("geometry_model.bin")
geom_model.saveToXML("geometry_model.xml", "geometry")

print(f"当前几何对象数量: {geom_model.ngeoms}")

object_id = geom_model.getGeometryId("sphere2")
print(f"几何对象 'sphere2' 的 ID: {object_id}")

