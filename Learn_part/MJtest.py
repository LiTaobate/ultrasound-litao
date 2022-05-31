import numpy as np 
from robosuite.models import MujocoWorldBase
world = MujocoWorldBase()

from robosuite.models.arenas import TableArena
# mujoco_arena = TableArena()

mujoco_arena = TableArena()
mujoco_arena.set_origin([0.8, 0, 0])#这里的set_origin是对所有对象应用恒定偏移。在X轴偏移0.8
world.merge(mujoco_arena)

model = world.get_model(mode="mujoco_py")
from mujoco_py import MjSim, MjViewer

sim = MjSim(model)#启动仿真
viewer = MjViewer(sim)#可视化仿真
viewer.vopt.geomgroup[0] = 0 # 禁用碰撞网格的可视化，就是不显示碰撞提醒（绿色的碰撞覆盖面）

for i in range(10000):
#   sim.data.ctrl[:] = 0
#   sim.data.ctrl[:6] = 1
  sim.step()
  viewer.render() #显示渲染过程
