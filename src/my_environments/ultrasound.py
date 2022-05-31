from collections import OrderedDict
import os
import numpy as np
import pandas as pd
import re
from klampt.model import trajectory
import roboticstoolbox as rtb

from spatialmath import SE3

from robosuite.utils.transform_utils import convert_quat, quat2mat, mat2euler
from robosuite.utils.mjcf_utils import CustomMaterial

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.observables import Observable, sensor
from robosuite.models.base import MujocoModel

import robosuite.utils.transform_utils as T

import sys
print(sys.path)
sys.path.append("/home/litao/Documents/robotic-ultrasound-imaging-master-1/src/my_models/objects")
sys.path.append("/home/litao/Documents/robotic-ultrasound-imaging-master-1/src/my_models/tasks")
sys.path.append("/home/litao/Documents/robotic-ultrasound-imaging-master-1/src/my_models/arenas")
sys.path.append("/home/litao/Documents/robotic-ultrasound-imaging-master-1/src/utils")
sys.path.append("/home/litao/Documents/robotic-ultrasound-imaging-master-1/src/my_models")
from xml_objects import SoftTorsoObject, BoxObject, SoftBoxObject
from ultrasound_task import UltrasoundTask
from ultrasound_arena import UltrasoundArena
from quaternion_1 import distance_quat, difference_quat  #  四元数


class Ultrasound(SingleArmEnv):
    """
    This class corresponds to the ultrasound task for a single robot arm.
    此类对应于单个机械臂的超声任务。
    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!
        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.
        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param
        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param
        table_full_size (3-tuple): x, y, and z dimensions of the table.
        table_friction (3-tuple): the three mujoco friction parameters for
            the table.
        use_camera_obs (bool): if True, every observation includes rendered image(s)
        use_object_obs (bool): if True, include object (cube) information in
            the observation.
        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.
        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.
        has_offscreen_renderer (bool): True if using off-screen rendering
        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse
        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.
        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.
        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).
        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.
        horizon (int): Every episode lasts for exactly @horizon timesteps.
        ignore_done (bool): True if never terminating the environment (ignore @horizon).
        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables
        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.
            :Note: At least one camera must be specified if @use_camera_obs is True.
            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).
        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.
        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.
        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.
        early_termination (bool): If True, episode is allowed to finish early.
        save_data (bool): If True, data from the episode is collected and saved.
        deterministic_trajectory (bool): If True, chooses a deterministic trajectory which goes along the x-axis of the torso.
        torso_solref_randomization (bool): If True, randomize the stiffness and damping parameter of the torso. 
        initial_probe_pos_randomization (bool): If True, Gaussian noise will be added to the initial position of the probe.
        use_box_torso (bool): If True, use a box shaped soft body. Else, use a cylinder shaped soft body.
    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",          #指定如何在环境中定位机器人（默认为“default”）。从对于大多数单臂环境，此参数对机器人设置没有影响。
        controller_configs=None,               #如果设置，包含相关的控制器参数，用于创建一个自定义控制器。否则，将默认控制器用于此特定任务。
        gripper_types="UltrasoundProbeGripper",#夹持器的类型，用于实例化。夹持器工厂的夹持器模型。默认为“default”，即关联的默认夹具与机器人一起使用“机器人”规范。None removes the gripper, and any other (valid) model
        initialization_noise="default",        
        table_full_size=(0.8, 0.8, 0.05),      #表格的 x、y 和 z 维度
        table_friction=100*(1., 5e-3, 1e-4),   #三个 mujoco 摩擦参数
        use_camera_obs=True,                   #如果为真，则每个观察都包括渲染图像
        use_object_obs=True,                   #如果为真，则在其中包含对象（立方体）信息
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,            #如果提供，将用于在每次重置时放置对象，否则为 UniformRandomSampler默认使用。
        has_renderer=False,                    #如果为真，则渲染模拟状态查看器而不是无头模式。
        has_offscreen_renderer=True,           #如果使用离屏渲染则为真
        render_camera="frontview",             #如果 `has_renderer` 为 True，则要渲染的相机名称。将此值设置为“无”
        render_collision_mesh=False,           #如果在相机中渲染碰撞网格，则为真。否则为假。
        render_visual_mesh=True,               #如果在相机中渲染视觉网格，则为真。否则为假。
        render_gpu_device_id=-1,               #对应于用于离屏渲染的 GPU 设备 ID.默认为 -1，在这种情况下，设备将从环境变量中推断出来
        control_freq=20,                       #每秒接收多少个控制信号。这设置了数量每个动作输入之间经过的模拟时间。
        horizon=1000,                          #每一集都持续精确的时间步长。
        ignore_done=False,                     #如果从不终止环境则为真
        hard_reset=True,                       #如果为真，则在重置调用时重新加载模型、模拟和渲染对象，否则,只调用 sim.reset 并重置所有 robosuite 内部变量
        camera_names="agentview",              #要渲染的相机的名称。
        camera_heights=256,                    #相机帧的高度。所有相机的帧都使用相同的高度，否则它应该是一个长度相同的列表
        camera_widths=256,                     #相机帧的宽度。所有相机的帧都使用相同的宽度，否则它应该是一个长度相同的列表
        camera_depths=False,                   #如果渲染 RGB-D 则为真，否则为 RGB。如果要对所有相机使用相同的深度设置，则设置为布尔值，否则它应该是长度相同的列表
        early_termination=False,               #如果为 True，则允许episode提前结束。
        save_data=False,                       #如果为 True，则收集并保存episode中的数据。
        deterministic_trajectory=False,        #如果为真，则选择沿着 tor 的 x 轴的确定性轨迹
        torso_solref_randomization=False,      #如果为 True，则随机化躯干的刚度和阻尼参数。
        initial_probe_pos_randomization=False, #如果为True，高斯噪声将被添加到探针的初始位置。
        use_box_torso=True,                    #如果为 True，则使用箱形软体。 否则，请使用圆柱形软体。
    ):
        assert gripper_types == "UltrasoundProbeGripper",\
            "Tried to specify gripper other than UltrasoundProbeGripper in Ultrasound environment!"

        assert robots == "UR5e" or robots == "Panda", \
            "Robot must be UR5e or Panda!"

        assert "OSC" or "HMFC" in controller_configs["type"], \
            "The robot controller must be of type OSC or HMFC"

        # settings for table top桌面设置
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # settings for joint initialization noise (Gaussian)初始化高斯噪声
        self.mu = 0
        self.sigma = 0.010

        # settings for contact force running mean 接触力运行平均值的设置
        self.alpha = 0.1    # 衰减因子 decay factor (high alpha -> discounts older observations faster). Must be in (0, 1)

        # reward configuration  reward设置
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # error multipliers 误差
        self.pos_error_mul = 90
        self.ori_error_mul = 0.2
        self.vel_error_mul = 45
        self.force_error_mul = 0.7
        self.der_force_error_mul = 0.01

        # reward multipliers 奖励
        self.pos_reward_mul = 5
        self.ori_reward_mul = 1
        self.vel_reward_mul = 1
        self.force_reward_mul = 3
        self.der_force_reward_mul = 2

        # desired states  期望
        self.goal_quat = np.array([-0.69192486,  0.72186726, -0.00514253, -0.01100909]) # Upright probe orientation found from experimenting (x,y,z,w)从实验中发现的直立探头方向 (x,y,z,w)
        # self.goal_velocity = 0.04                   # norm of velocity vector 速度矢量的范数
        self.goal_velocity = 0.1
        self.goal_contact_z_force = 5               # (N)  
        self.goal_der_contact_z_force = 0           # derivative of contact force   接触力微分

        # early termination configuration  提前终止配置
        self.pos_error_threshold = 1.0
        self.ori_error_threshold = 0.10

        # examination trajectory  超声检查轨迹
        self.top_torso_offset = 0.039 if use_box_torso else 0.041      # offset from z_center of torso to top of torso 从躯干 z_center 到躯干顶部的偏移量
        self.x_range = 0.15                                 #   躯干在 x 方向上从中心到末端有多大
        self.y_range = 0.09 if use_box_torso else 0.05      #   躯干在 y 方向上从中心到末端有多大
        self.grid_pts = 50                                  #   网格中有多少个点
                                            
        # whether to use ground-truth object states 是否使用真实对象状态
        self.use_object_obs = use_object_obs

        # object placement initializer 对象放置初始值设定项
        self.placement_initializer = placement_initializer

        # randomization settings 刚度和阻尼随机化设置
        self.torso_solref_randomization = torso_solref_randomization
        self.initial_probe_pos_randomization = initial_probe_pos_randomization

        # misc settings 其他设置
        self.early_termination = early_termination
        self.save_data = save_data
        self.deterministic_trajectory = deterministic_trajectory
        self.use_box_torso = use_box_torso

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=None,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
        )
        

    def reward(self, action=None):
        """
        Reward function for the task.

        Args:参数
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """

        reward = 0.

        ee_current_ori = convert_quat(self._eef_xquat, to="wxyz")   # (w, x, y, z) 四元数
        ee_desired_ori = convert_quat(self.goal_quat, to="wxyz")

        # position 四元数
        self.pos_error = np.square(self.pos_error_mul * (self._eef_xpos[0:-1] - self.traj_pt[0:-1]))
        self.pos_reward = self.pos_reward_mul * np.exp(-1 * np.linalg.norm(self.pos_error))

        # orientation 方向
        self.ori_error = self.ori_error_mul * distance_quat(ee_current_ori, ee_desired_ori)
        self.ori_reward = self.ori_reward_mul * np.exp(-1 * self.ori_error)

        # velocity 速度
        self.vel_error =  np.square(self.vel_error_mul * (self.vel_running_mean - self.goal_velocity))
        self.vel_reward = self.vel_reward_mul * np.exp(-1 * np.linalg.norm(self.vel_error))
        
        # force  接触力
        self.force_error = np.square(self.force_error_mul * (self.z_contact_force_running_mean - self.goal_contact_z_force))
        self.force_reward = self.force_reward_mul * np.exp(-1 * self.force_error) if self._check_probe_contact_with_torso() else 0

        # derivative force 接触力导数
        self.der_force_error = np.square(self.der_force_error_mul * (self.der_z_contact_force - self.goal_der_contact_z_force))
        self.der_force_reward = self.der_force_reward_mul * np.exp(-1 * self.der_force_error) if self._check_probe_contact_with_torso() else 0

        # add rewards 添加奖励
        reward += (self.pos_reward + self.ori_reward + self.vel_reward + self.force_reward + self.der_force_reward)

        return reward


    def _load_model(self):
        """
        加载一个xml模型，把它放到self.model中
        """
        super()._load_model()

        # Adjust base pose accordingly相应地调整基础姿势
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # Load model for table top workspace桌面工作区的加载模型
        mujoco_arena = UltrasoundArena()

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Initialize torso object始化躯干对象
        #调用错误tao
        
        self.torso = SoftBoxObject(name="torso") if self.use_box_torso else SoftTorsoObject(name="torso")
        # self.torso = SoftBoxObject() if self.use_box_torso else SoftTorsoObject()
        if self.torso_solref_randomization:
            # Randomize torso's stiffness and damping (values are taken from my project thesis)随机化躯干的刚度和阻尼
            stiffness = np.random.randint(1300, 1600)
            damping = np.random.randint(17, 41)

            self.torso.set_damping(damping)
            self.torso.set_stiffness(stiffness)

        # Create placement initializer创建初始化位置
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.torso)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=[self.torso],
                x_range=[0, 0], #[-0.12, 0.12],
                y_range=[0, 0], #[-0.12, 0.12],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.005,
            )

        # task includes arena, robot, and objects of interest任务包括舞台、机器人和感兴趣的对象
        self.model = UltrasoundTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots], 
            mujoco_objects=[self.torso]
        )


    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        设置对重要组件的引用。 引用通常是
        索引或指向相应元素的索引列表
        在一个扁平数组中，这就是 MuJoCo 存储物理模拟数据的方式。
        """
        super()._setup_references()

        # additional object references from this env此环境中的其他对象引用
        self.torso_body_id = self.sim.model.body_name2id(self.torso.root_body)
        self.probe_id = self.sim.model.body_name2id(self.robots[0].gripper.root_body)
        

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
            设置用于此环境的可观察对象。 如果启用，则创建基于对象的可观察对象

         返回:
             OrderedDict:将可观察名称映射到其对应的可观察对象的字典
        """
        observables = super()._setup_observables()

        pf = self.robots[0].robot_model.naming_prefix

        # Remove unnecessary observables 删除不必要的 observables
        del observables[pf + "joint_pos"]
        del observables[pf + "joint_pos_cos"]
        del observables[pf + "joint_pos_sin"]
        del observables[pf + "joint_vel"]
        del observables[pf + "gripper_qvel"]
        del observables[pf + "gripper_qpos"]
        del observables[pf + "eef_pos"]
        del observables[pf + "eef_quat"]

        sensors = []

        # probe information超声探头信息
        modality = f"{pf}proprio"       # Need to use this modality since proprio obs cannot be empty in GymWrapper
        #需要使用这种模式，因为 GymWrapper 中的 proprio obs 不能为空

        @sensor(modality=modality)
        def eef_contact_force(obs_cache):
            return self.sim.data.cfrc_ext[self.probe_id][-3:]

        @sensor(modality=modality)
        def eef_torque(obs_cache):
            return self.robots[0].ee_torque

        @sensor(modality=modality)
        def eef_vel(obs_cache):
            return self.robots[0]._hand_vel

        @sensor(modality=modality)
        def eef_contact_force_z_diff(obs_cache):
            return self.z_contact_force_running_mean - self.goal_contact_z_force

        @sensor(modality=modality)
        def eef_contact_derivative_force_z_diff(obs_cache):
            return self.der_z_contact_force - self.goal_der_contact_z_force

        @sensor(modality=modality)
        def eef_vel_diff(obs_cache):
            return self.vel_running_mean - self.goal_velocity

        @sensor(modality=modality)
        def eef_pose_diff(obs_cache):
            pos_error = self._eef_xpos - self.traj_pt
            quat_error = difference_quat(self._eef_xquat, self.goal_quat)
            pose_error = np.concatenate((pos_error, quat_error))
            return pose_error

        sensors += [
            eef_contact_force,
            eef_torque, 
            eef_vel, 
            eef_contact_force_z_diff, 
            eef_contact_derivative_force_z_diff, 
            eef_vel_diff, 
            eef_pose_diff]

        names = [s.__name__ for s in sensors]

        # Create observables创建可观察对象
        for name, s in zip(names, sensors):
            observables[name] = Observable(
                name=name,
                sensor=s,
                sampling_rate=self.control_freq,
            )

        return observables


    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        重置模拟内部配置。
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        #如果我们不是直接从xml加载，使用初始化采样器重置所有对象的位置
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects所有对象的放置初始化程序示例
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions 循环遍历所有对象并重置它们的位置
            for obj_pos, _, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array([0.5, 0.5, -0.5, -0.5])]))
                self.sim.forward()      # update模拟状态
                
        # says if probe has been in touch with torso表示探头是否与躯干接触
        self.has_touched_torso = False

        # initial position of end-effector 末端执行器初始位置
        self.ee_initial_pos = self._eef_xpos

        # create trajectory 创建轨迹
        self.trajectory = self.get_trajectory()
        
        # initialize trajectory step 初始化轨迹步骤
        self.initial_traj_step = np.random.default_rng().uniform(low=0, high=self.num_waypoints - 1)
        self.traj_step = self.initial_traj_step                                    # 评估轨迹的步骤。必须在区间 [0, num_waypoints - 1]
        
        # set first trajectory point 设置第一个轨迹点
        self.traj_pt = self.trajectory.eval(self.traj_step)
        self.traj_pt_vel = self.trajectory.deriv(self.traj_step)

        # give controller access to robot (and its measurements)允许控制器访问机器人（及其测量值）
        if self.robots[0].controller.name == "HMFC":
            self.robots[0].controller.set_robot(self.robots[0])

        # initialize controller's trajectory 初始化控制器的轨迹
        self.robots[0].controller.traj_pos = self.traj_pt
        self.robots[0].controller.traj_ori = T.quat2axisangle(self.goal_quat)

        # get initial joint positions for robot 获取机器人的初始关节位置
        init_qpos = self._get_initial_qpos()

        # override initial robot joint positions 覆盖最初的机器人关节位置
        self.robots[0].set_robot_joint_positions(init_qpos)

        # update controller with new initial joints 使用新的初始关节更新控制器
        self.robots[0].controller.update_initial_joints(init_qpos)

        # initialize previously contact force measurement 初始化先前的接触力测量
        self.prev_z_contact_force = 0 

        # intialize derivative of contact force 初始化接触力的导数
        self.der_z_contact_force = 0
        
        # initialize running mean of velocity  初始化速度的运行平均值
        self.vel_running_mean = np.linalg.norm(self.robots[0]._hand_vel)
 
        # initialize running mean of contact force 初始化接触力的运行平均值
        self.z_contact_force_running_mean = self.sim.data.cfrc_ext[self.probe_id][-1]
 
        # initialize data collection 初始化数据收集
        if self.save_data:
            # simulation data
            self.data_ee_pos = np.array(np.zeros((self.horizon, 3)))
            self.data_ee_goal_pos = np.array(np.zeros((self.horizon, 3)))
            self.data_ee_ori_diff = np.array(np.zeros(self.horizon))
            self.data_ee_vel = np.array(np.zeros((self.horizon, 3)))
            self.data_ee_goal_vel = np.array(np.zeros(self.horizon))
            self.data_ee_running_mean_vel = np.array(np.zeros(self.horizon))
            self.data_ee_quat = np.array(np.zeros((self.horizon, 4)))               # (x,y,z,w)
            self.data_ee_goal_quat = np.array(np.zeros((self.horizon, 4)))          # (x,y,z,w)
            self.data_ee_diff_quat = np.array(np.zeros(self.horizon))               # (x,y,z,w)
            self.data_ee_z_contact_force = np.array(np.zeros(self.horizon))
            self.data_ee_z_goal_contact_force = np.array(np.zeros(self.horizon))
            self.data_ee_z_running_mean_contact_force = np.array(np.zeros(self.horizon))
            self.data_ee_z_derivative_contact_force = np.array(np.zeros(self.horizon))
            self.data_ee_z_goal_derivative_contact_force = np.array(np.zeros(self.horizon))
            self.data_is_contact = np.array(np.zeros(self.horizon))
            self.data_q_pos = np.array(np.zeros((self.horizon, self.robots[0].dof)))
            self.data_q_torques = np.array(np.zeros((self.horizon, self.robots[0].dof)))
            self.data_time = np.array(np.zeros(self.horizon))

            # reward data
            self.data_pos_reward = np.array(np.zeros(self.horizon))
            self.data_ori_reward = np.array(np.zeros(self.horizon))
            self.data_vel_reward = np.array(np.zeros(self.horizon))
            self.data_force_reward = np.array(np.zeros(self.horizon))
            self.data_der_force_reward = np.array(np.zeros(self.horizon))

            # policy/controller data
            self.data_action = np.array(np.zeros((self.horizon, self.robots[0].action_dim)))


    def _post_action(self, action):
        """
        In addition to super method, add additional info if requested

        Args:
            action (np.array): Action to execute within the environment

        Returns:
            3-tuple:
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) info about current env step
        除了 super 方法，如果需要，添加其他信息
         参数：
             action (np.array)：在环境中执行的动作
         回报：
             三元组：
                 - （浮动）环境奖励
                 - (bool) 当前剧集是否完成
                 - （dict）有关当前环境步骤的信息
        """
        reward, done, info = super()._post_action(action)
 
        # Convert to trajectory timstep 转换为轨迹时间步长
        normalizer = (self.horizon / (self.num_waypoints - 1))                  # equally many timesteps to reach each waypoint 到达每个航路点的时间步长相等
        self.traj_step = self.timestep / normalizer + self.initial_traj_step

        # update trajectory point 更新轨迹点
        self.traj_pt = self.trajectory.eval(self.traj_step)

        # update controller's trajectory 更新控制器的轨迹
        self.robots[0].controller.traj_pos = self.traj_pt
 
        # update velocity running mean (simple moving average) 更新速度运行平均值（简单移动平均）
        self.vel_running_mean += ((np.linalg.norm(self.robots[0]._hand_vel) - self.vel_running_mean) / self.timestep)

        # update derivative of contact force 更新接触力的导数
        z_contact_force = self.sim.data.cfrc_ext[self.probe_id][-1]
        self.der_z_contact_force = (z_contact_force - self.prev_z_contact_force) / self.control_timestep
        self.prev_z_contact_force = z_contact_force

        # update contact force running mean (exponential moving average) 更新接触力运行平均值（指数移动平均值）
        self.z_contact_force_running_mean = self.alpha * z_contact_force + (1 - self.alpha) * self.z_contact_force_running_mean

        # check for early termination 检查提前终止
        if self.early_termination: 
            done = done or self._check_terminated()

        # collect data 收集数据
        if self.save_data:
            # simulation data 仿真数据
            self.data_ee_pos[self.timestep - 1] = self._eef_xpos
            self.data_ee_goal_pos[self.timestep - 1] = self.traj_pt
            self.data_ee_vel[self.timestep - 1] = self.robots[0]._hand_vel
            self.data_ee_goal_vel[self.timestep - 1] = self.goal_velocity
            self.data_ee_running_mean_vel[self.timestep -1] = self.vel_running_mean
            self.data_ee_quat[self.timestep - 1] = self._eef_xquat
            self.data_ee_goal_quat[self.timestep - 1] = self.goal_quat
            self.data_ee_diff_quat[self.timestep - 1] = distance_quat(convert_quat(self._eef_xquat, to="wxyz"), convert_quat(self.goal_quat, to="wxyz"))
            self.data_ee_z_contact_force[self.timestep - 1] = self.sim.data.cfrc_ext[self.probe_id][-1]
            self.data_ee_z_goal_contact_force[self.timestep - 1] = self.goal_contact_z_force
            self.data_ee_z_running_mean_contact_force[self.timestep - 1] = self.z_contact_force_running_mean
            self.data_ee_z_derivative_contact_force[self.timestep - 1] = self.der_z_contact_force
            self.data_ee_z_goal_derivative_contact_force[self.timestep - 1] = self.goal_der_contact_z_force
            self.data_is_contact[self.timestep - 1] = self._check_probe_contact_with_torso()
            self.data_q_pos[self.timestep - 1] = self.robots[0]._joint_positions
            self.data_q_torques[self.timestep - 1] = self.robots[0].torques
            self.data_time[self.timestep - 1] = (self.timestep - 1) / self.horizon * 100                         # percentage of completed episode完成100%

            # reward data 
            self.data_pos_reward[self.timestep - 1] = self.pos_reward
            self.data_ori_reward[self.timestep - 1] = self.ori_reward
            self.data_vel_reward[self.timestep - 1] = self.vel_reward
            self.data_force_reward[self.timestep - 1] = self.force_reward
            self.data_der_force_reward[self.timestep - 1] = self.der_force_reward

            # policy/controller data
            self.data_action[self.timestep - 1] = action
        
        # save data 保存数据
        if done and self.save_data:
            # simulation data
            sim_data_fldr = "simulation_data"
            self._save_data(self.data_ee_pos, sim_data_fldr, "ee_pos")
            self._save_data(self.data_ee_goal_pos, sim_data_fldr, "ee_goal_pos")
            self._save_data(self.data_ee_vel, sim_data_fldr, "ee_vel")
            self._save_data(self.data_ee_goal_vel, sim_data_fldr, "ee_goal_vel")
            self._save_data(self.data_ee_running_mean_vel, sim_data_fldr, "ee_running_mean_vel")
            self._save_data(self.data_ee_quat, sim_data_fldr, "ee_quat")
            self._save_data(self.data_ee_goal_quat, sim_data_fldr, "ee_goal_quat")
            self._save_data(self.data_ee_diff_quat, sim_data_fldr, "ee_diff_quat")
            self._save_data(self.data_ee_z_contact_force, sim_data_fldr, "ee_z_contact_force")
            self._save_data(self.data_ee_z_goal_contact_force, sim_data_fldr, "ee_z_goal_contact_force")
            self._save_data(self.data_ee_z_running_mean_contact_force, sim_data_fldr, "ee_z_running_mean_contact_force")
            self._save_data(self.data_ee_z_derivative_contact_force, sim_data_fldr, "ee_z_derivative_contact_force")
            self._save_data(self.data_ee_z_goal_derivative_contact_force, sim_data_fldr, "ee_z_goal_derivative_contact_force")
            self._save_data(self.data_is_contact, sim_data_fldr, "is_contact")
            self._save_data(self.data_q_pos, sim_data_fldr, "q_pos")
            self._save_data(self.data_q_torques, sim_data_fldr, "q_torques")
            self._save_data(self.data_time, sim_data_fldr, "time")

            # reward data
            reward_data_fdlr = "reward_data"
            self._save_data(self.data_pos_reward, reward_data_fdlr, "pos")
            self._save_data(self.data_ori_reward, reward_data_fdlr, "ori")
            self._save_data(self.data_vel_reward, reward_data_fdlr, "vel")
            self._save_data(self.data_force_reward, reward_data_fdlr, "force")
            self._save_data(self.data_der_force_reward, reward_data_fdlr, "derivative_force")

            # policy/controller data
            self._save_data(self.data_action, "policy_data", "action")


        return reward, done, info


    def visualize(self, vis_settings):
        """
        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        参数：
             vis_settings(dict)：映射到T/F的可视化关键字，决定是否应该可视化那个特定的组件。 应具有“grippers”关键字以及指定的任何其他相关选项。
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)


    def _check_success(self):
        return False


    def _check_terminated(self):
        """
        Check if the task has completed one way or another. The following conditions lead to termination:
            - Collision with table
            - Joint Limit reached
            - Deviates from trajectory position
            - Deviates from desired orientation when in contact with torso
            - Loses contact with torso

        Returns:
            bool: True if episode is terminated
        检查任务是否以一种或另一种方式完成。 以下情况会导致终止：
            - 与桌子碰撞
            - 达到关节极限
            - 偏离轨迹位置
            - 与躯干接触时偏离所需方向
            - 失去与躯干的接触

        回报：
            bool：如果剧集终止则为真
        """

        terminated = False

        # Prematurely terminate if reaching joint limits 达到关节极限时提前终止
        if self.robots[0].check_q_limits():
            print(40 * '-' + " JOINT LIMIT " + 40 * '-')
            terminated = True

        # Prematurely terminate if probe deviates away from trajectory (represented by a low position reward)
        #如果探针偏离轨迹，则提前终止（由低位置奖励表示）
        if np.linalg.norm(self.pos_error) > self.pos_error_threshold:
            print(40 * '-' + " DEVIATES FROM TRAJECTORY " + 40 * '-')
            terminated = True

        # Prematurely terminate if probe deviates from desired orientation when touching probe
        #如果在接触探头时探头偏离所需方向，则提前终止
        if self._check_probe_contact_with_torso() and self.ori_error > self.ori_error_threshold:
            print(40 * '-' + " (TOUCHING BODY) PROBE DEVIATES FROM DESIRED ORIENTATION " + 40 * '-')
            terminated = True

        # Prematurely terminate if probe loses contact with torso
        #如果探头与躯干失去接触，则提前终止
        if self.has_touched_torso and not self._check_probe_contact_with_torso():
            print(40 * '-' + " LOST CONTACT WITH TORSO " + 40 * '-')
            terminated = True

        return terminated


    def _get_contacts_objects(self, model):
        """
        Checks for any contacts with @model (as defined by @model's contact_geoms) and returns the set of
        contact objects currently in contact with that model (excluding the geoms that are part of the model itself).

        Args:
            model (MujocoModel): Model to check contacts for.

        Returns:
            set: Unique contact objects containing information about contacts with this model.

        Raises:
            AssertionError: [Invalid input type]
        检查与模型的的任何接触（由@model 的contact_geoms 定义）并返回当前与该模型接触的接触对象集（不包括模型本身的几何图形）。
        参数：
            模型（MujocoModel）：用于检查联系人的模型。
        回报：
            set：唯一的联系人对象，包含有关此模型的联系人的信息。
        加注：
            AssertionError：[无效的输入类型]
        """
        # Make sure model is MujocoModel type
        assert isinstance(model, MujocoModel), \
            "Inputted model must be of type MujocoModel; got type {} instead!".format(type(model))
        contact_set = set()
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            # check contact geom in geoms; add to contact set if match is found 
            #在geoms中检查接触geom； 如果找到匹配项，则添加到contact set
            g1, g2 = self.sim.model.geom_id2name(contact.geom1), self.sim.model.geom_id2name(contact.geom2)
            if g1 in model.contact_geoms or g2 in model.contact_geoms:
                contact_set.add(contact)

        return contact_set


    def _check_probe_contact_with_upper_part_torso(self):
        """
        Check if the probe is in contact with the upper/top part of torso. Touching the torso on the sides should not count as contact.
        Returns:
            bool: True if probe both is in contact with upper part of torso and inside distance threshold from the torso center.
        检查探头是否与躯干的上部/顶部接触。 触摸模型两侧的躯干不应算作接触。
        返回:
            bool：如果探头都与躯干的上部接触并且距离躯干中心的距离阈值内，则为真。
        """     
        # check for contact only if probe is in contact with upper part and close to torso center
        #仅当探头与上部接触并靠近躯干中心时检查接触
        if  self._eef_xpos[-1] >= self._torso_xpos[-1] and np.linalg.norm(self._eef_xpos[:2] - self._torso_xpos[:2]) < 0.14:
            return self._check_probe_contact_with_torso()

        return False


    def _check_probe_contact_with_torso(self):
        """
        Check if the probe is in contact with the torso.
        NOTE This method utilizes the autogenerated geom names for MuJoCo-native composite objects
        Returns:
            bool: True if probe is in contact with torso
        检查探头是否与躯干接触。
        注意此方法利用 MuJoCo-native 复合对象的自动生成的几何名称
        返回:
             bool：如果探针与躯干接触，则为真
        """     
        gripper_contacts = self._get_contacts_objects(self.robots[0].gripper)
        reg_ex = "[G]\d+[_]\d+[_]\d+$"

        # check contact with torso geoms based on autogenerated names
        #根据自动生成的名称检查与躯干几何的联系
        for contact in gripper_contacts:
            g1, g2 = self.sim.model.geom_id2name(contact.geom1), self.sim.model.geom_id2name(contact.geom2) 
            match1 = re.search(reg_ex, g1)
            match2 = re.search(reg_ex, g2)
            if match1 != None or match2 != None:
                contact_normal_axis = contact.frame[:3]
                self.has_touched_torso = True
                return True
    
        return False

    
    def _check_probe_contact_with_table(self):
        """
        Check if the probe is in contact with the tabletop.

        Returns:
            bool: True if probe is in contact with table
        检查探头是否与桌面接触。
        返回:
            bool：如果探针与桌子接触，则为真
        """
        return self.check_contact(self.robots[0].gripper, "table_collision")
    

    def get_trajectory(self):
        """
        Calculates a trajectory between two waypoints on the torso. The waypoints are extracted from a grid on the torso.
        The first waypoint is given at time t=0, and the second waypoint is given at t=1.
        If self.deterministic_trajectory is true, a deterministic trajectory along the x-axis of the torso is calculated.

        Args:

        Returns:
            (klampt.model.trajectory Object):  trajectory

        计算躯干上两个航点之间的轨迹。 航点是从躯干上的网格中提取的。
         第一个航点在时间 t=0 给出，第二个航点在 t=1 给出。
         如果 self.deterministic_trajectory 为真，则计算沿躯干 x 轴的确定性轨迹。
         参数：
         返回:
            (klampt.model.trajectory Object): 轨迹
        """
        grid = self._get_torso_grid()

        if self.deterministic_trajectory:
            start_point = [0.062, -0.020,  0.896]
            end_point = [-0.032, -0.075,  0.896]

            #start_point = [grid[0, 0], grid[1, 4], self._torso_xpos[-1] + self.top_torso_offset]
            #end_point = [grid[0, int(self.grid_pts / 2) - 1], grid[1, 5], self._torso_xpos[-1] + self.top_torso_offset]
        else:   
            start_point = self._get_waypoint(grid)
            end_point = self._get_waypoint(grid)
        
        milestones = np.array([start_point, end_point])
        self.num_waypoints = np.size(milestones, 0)

        return trajectory.Trajectory(milestones=milestones)


    def _get_torso_grid(self):
        """
        Creates a 2D grid in the xy-plane on the top of the torso.

        Args:

        Returns:
            (numpy.array):  grid. First row contains x-coordinates and the second row contains y-coordinates.
        在躯干顶部的 xy 平面中创建 2D 网格。 参数：返回：（numpy.array）：网格。 第一行包含 x 坐标，第二行包含 y 坐标。
        """
        x = np.linspace(-self.x_range + self._torso_xpos[0] + 0.03, self.x_range + self._torso_xpos[0], num=self.grid_pts)  # 由于靠近机器人底座的奇怪机器人角度，在负范围内添加偏移量add offset in negative range due to weird robot angles close to robot base
        y = np.linspace(-self.y_range + self._torso_xpos[1], self.y_range + self._torso_xpos[1], num=self.grid_pts)

        x = np.array([x])
        y = np.array([y])

        return np.concatenate((x, y))

    
    def _get_waypoint(self, grid):
        """
        Extracts a random waypoint from the grid.从网格中提取随机航路点。
        Args:
        Returns:
            (numpy.array):  waypoint
        """
        x_pos = np.random.choice(grid[0])
        y_pos = np.random.choice(grid[1])
        z_pos = self._torso_xpos[-1] + self.top_torso_offset

        return np.array([x_pos, y_pos, z_pos])
        
    
    def _get_initial_qpos(self):
        """
        Calculates the initial joint position for the robot based on the initial desired pose (self.traj_pt, self.goal_quat).
        根据初始所需姿势（self.traj_pt、self.goal_quat）计算机器人的初始关节位置。
        If self.initial_probe_pos_randomization is True, Guassian noise is added to the initial position of the probe.
        如果 self.initial_probe_pos_randomization 为 True，则将高斯噪声添加到探针的初始位置。
        Args:

        Returns:
            (np.array): n joint positions 
        """
        pos = np.array(self.traj_pt)
        if self.initial_probe_pos_randomization:
            pos = self._add_noise_to_pos(pos)

        pos = self._convert_robosuite_to_toolbox_xpos(pos)
        ori_euler = mat2euler(quat2mat(self.goal_quat))

        # desired pose 想要的姿势
        T = SE3(pos) * SE3.RPY(ori_euler)

        # find initial joint positions 找到初始关节位置
        if self.robots[0].name == "UR5e":
            robot = rtb.models.DH.UR5()
            sol = robot.ikine_min(T, q0=self.robots[0].init_qpos)

            # flip last joint around (pi) 翻转最后一个关节 (pi)
            sol.q[-1] -= np.pi
            return sol.q

        elif self.robots[0].name == "Panda":
            robot = rtb.models.DH.Panda()
            sol = robot.ikine_min(T, q0=self.robots[0].init_qpos)
            return sol.q


    def _convert_robosuite_to_toolbox_xpos(self, pos):
        """
        Converts origin used in robosuite to origin used for robotics toolbox. Also transforms robosuite world frame (vectors x, y, z) to
        to correspond to world frame used in toolbox.将 robosuite 中使用的原点转换为机器人工具箱中使用的原点。 还将机器人套件世界框架（向量 x、y、z）
        转换为对应于工具箱中使用的世界框架。

        Args:
            pos (np.array): position (x,y,z) given in robosuite coordinates and frame 在robosuite坐标和框架中给出的位置 (x,y,z)

        Returns:
            (np.array):  position (x,y,z) given in robotics toolbox coordinates and frame 在机器人工具箱坐标和框架中给出的位置 (x,y,z)
        """
        xpos_offset = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])[0]
        zpos_offset = self.robots[0].robot_model.top_offset[-1] - 0.016

        # the numeric offset values have been found empirically, where they are chosen so that 数字偏移值是根据经验找到的，它们的选择使得
        # self._eef_xpos matches the desired position.匹配所需位置。
        if self.robots[0].name == "UR5e":
            return np.array([-pos[0] + xpos_offset + 0.08, -pos[1] + 0.025, pos[2] - zpos_offset + 0.15]) 

        if self.robots[0].name == "Panda":
            return np.array([pos[0] - xpos_offset - 0.06, pos[1], pos[2] - zpos_offset + 0.111])


    def _add_noise_to_pos(self, init_pos):
        """
        Adds Gaussian noise (variance) to the position.将高斯噪声（方差）添加到位置。

        Args:
            init_pos (np.array): initial probe position 初始探头位置

        Returns:
            (np.array):  position with added noise 添加噪声的位置
        """
        z_noise = np.random.normal(self.mu, self.sigma, 1)
        xy_noise = np.random.normal(self.mu, self.sigma/4, 2)

        x = init_pos[0] + xy_noise[0]
        y = init_pos[1] + xy_noise[1]
        z = init_pos[2] + z_noise[0]

        return np.array([x, y, z])


    def _save_data(self, data, fldr, filename):
        """
        Saves data to desired path.数据保存路径

        Args:
            data (np.array): Data to be saved 
            fldr (string): Name of destination folder
            filename (string): Name of file

        Returns:
        """
        os.makedirs(fldr, exist_ok=True)

        idx = 1
        path = os.path.join(fldr, filename + "_" + str(idx) + ".csv")

        while os.path.exists(path):
            idx += 1
            path = os.path.join(fldr, filename + "_" + str(idx) + ".csv")

        pd.DataFrame(data).to_csv(path, header=None, index=None)


    @property
    def _torso_xpos(self):
        """
        Grabs torso center position找到人体模型的中心
        Returns:
            np.array: torso pos (x,y,z)
        """
        return np.array(self.sim.data.body_xpos[self.torso_body_id]) 