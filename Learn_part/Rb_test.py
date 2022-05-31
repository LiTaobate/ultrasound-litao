import numpy as np
import robosuite as suite

# create environment instance创建环境实例
env = suite.make(
    # env_name="Lift", # try with other tasks like "Stack" and "Door"
    env_name="Lift", 
    robots="UR5e",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    # use_camera_obs=True,
)

# 重置环境
env.reset()

for i in range(1000):
    action = np.random.randn(env.robots[0].dof) # 设置随机运动
    obs, reward, done, info = env.step(action)  # 在环境中采取行动
    env.render()  #渲染展示render：渲染