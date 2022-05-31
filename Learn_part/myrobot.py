#第一步：设置世界，采用自带的默认世界MujocoWorldBase
import numpy as np 
from robosuite.models import MujocoWorldBase
world = MujocoWorldBase()

#第二步：创建自己的机器人
from robosuite.models.robots import UR5e
# mujoco_robot1 = Panda()
mujoco_robot1 = UR5e()

#我们可以通过创建一个抓手实例并在机器人上调用 add_gripper 方法来为机器人添加一个抓手。
from robosuite.models.grippers import gripper_factory
gripper = gripper_factory('PandaGripper')
mujoco_robot1.add_gripper(gripper)

#要将机器人添加到世界中，我们将机器人放置到所需位置并将其合并到世界中
mujoco_robot1.set_base_xpos([0.5,0,0.8])#刚好能放置在桌子上，桌子高度为0.8
world.merge(mujoco_robot1)
#第三步：创建桌子。我们可以初始化创建桌子和地平面,TableArena代表的是一个拥有桌子的整体环境
from robosuite.models.arenas import TableArena

mujoco_arena = TableArena()
mujoco_arena.set_origin([0.8, 0, 0])#这里的set_origin是对所有对象应用恒定偏移。在X轴偏移0.8
world.merge(mujoco_arena)

#第四步：添加对象。创建一个球并将其添加到世界中。

from robosuite.models.objects import BallObject
sphere1 = BallObject(
    name="sphere1",
    size=[0.04],
    rgba=[0, 0.5, 0.5, 1]).get_obj()
sphere1.set('pos', '1.3 0 1.0')
world.worldbody.append(sphere1)

from robosuite.models.objects import BoxObject
sphere2 = BoxObject(
    name = "sphere2",
    size = [0.07, 0.07, 0.07],
    rgba=[0, 0.5, 0.5, 1]).get_obj()
sphere2.set('pos', '1.4 0 1.0')
world.worldbody.append(sphere2)

from robosuite.models.objects import CapsuleObject
sphere3 = CapsuleObject(
    name = "sphere3",
    size = [0.07,  0.07],
    rgba=[0.5, 0.5, 0.5, 1]).get_obj()#颜色(三个0.5是黑色，A值代表透明度)
sphere3.set('pos', '1.0 0 1.0')
world.worldbody.append(sphere3)


from robosuite.utils.mjcf_utils import new_joint


#第5步：运行模拟。一旦我们创建了对象，我们可以通过运行mujoco_py获得一个模型
model = world.get_model(mode="mujoco_py")
from mujoco_py import MjSim, MjViewer

sim = MjSim(model)#启动仿真
viewer = MjViewer(sim)#可视化仿真
viewer.vopt.geomgroup[0] = 0 # 禁用碰撞网格的可视化，就是不显示碰撞提醒（绿色的碰撞覆盖面）

for i in range(10000):
  sim.data.ctrl[:] = 0
#   sim.data.ctrl[:6] = 1
  sim.step()
  viewer.render() #显示渲染过程

