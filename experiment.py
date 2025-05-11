import pinocchio as pin
from pinocchio import GeometryObject, SE3


# 创建 GeometryModel 并添加对象
model = pin.GeometryModel()

print(model.existGeometryName("warner"))
