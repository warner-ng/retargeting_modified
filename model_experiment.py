import pinocchio as pin

model = pin.Model()

# 定义一个旋转关节（Revolute Joint）
joint_model = pin.JointModelRX()  # 例如，绕 X 轴旋转的关节
joint_placement = pin.SE3.Identity()  # 关节相对于父关节的位姿
parent_id = 0  # 父关节的索引
joint_name = "joint_0"

joint_id_0 = model.addJoint(parent_id, joint_model, joint_placement, joint_name)
print("Added joint with ID:", joint_id_0)

# 定义一个身体的惯性
body_inertia = pin.Inertia.FromBox(mass=1,
                                   length_x=1,
                                   length_y=1,
                                   length_z=1)


# 定义身体相对于关节的位姿
body_placement = pin.SE3.Identity()

# 将身体添加到关节上
model.appendBodyToJoint(joint_id_0, body_inertia, body_placement)

joint_id_0 = model.getJointId("joint_0")
body_id = model.getBodyId("body_frame")
print("Joint ID:", joint_id_0)
print("Body ID:", body_id)

data = model.createData()
is_consistent = model.check(data)
print("Model is consistent:", is_consistent)

# # 保存
# model.saveToBinary("model.bin")
# loaded_model = pin.Model()
# loaded_model.loadFromBinary("model.bin")
# model.saveToText("model.txt")

# 访问属性
print("Number of joints:", model.njoints) # 什么都不加的时候也是1
print("Number of bodies:", model.nbodies) # 什么都不加的时候也是1
print("Gravity vector:", model.gravity)





# 添加第二个关节 joint_1
joint_model = pin.JointModelRX()  # 绕 X 轴旋转的关节
joint_1_placement = pin.SE3.Identity()
joint_id_1 = model.addJoint(0, joint_model, joint_1_placement, "joint_1")


# 访问属性
print("添加新关节")
print("Number of joints:", model.njoints) # 什么都不加的时候也是1
print("Number of bodies:", model.nbodies) # 什么都不加的时候也是1
print("Gravity vector:", model.gravity)

body_inertia = pin.Inertia.FromSphere(mass=1,
                                      radius=1)
body_placement = pin.SE3.Identity()
model.appendBodyToJoint(joint_id_1, body_inertia, body_placement)

# 访问属性
print("添加新body")
print("Number of joints:", model.njoints) # 什么都不加的时候也是1
print("Number of bodies:", model.nbodies) # 什么都不加的时候也是1
print("Gravity vector:", model.gravity)



# add a body to the frame tree 到joint_1
frame_id_1 = model.addBodyFrame("body_0", joint_id_1, body_placement, 0)
frame_id_2 = model.addJointFrame(joint_id_1)

if model.existJointName("joint_0"):
    print("Joint exists.")
if model.existJointName("joint_1"):
    print("Joint exists.")
if model.existJointName("joint_2"):
    print("Joint exists.")
    
if model.existFrame("frame_0"):
    print("Frame exists.")

limits = model.hasConfigurationLimit()
print(limits)


modelB = pin.Model()
new_model = pin.appendModel(model,
                            modelB,
                            frame_in_modelA = frame_id_1,
                            aMb = joint_placement)





























