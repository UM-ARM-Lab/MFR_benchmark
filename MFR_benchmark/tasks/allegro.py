import numpy as np
import math
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import torch

# pytorch3d uses real part first for quaternions
# isaac gym uses real part last for quaternions
import pathlib
import pytorch_kinematics as pk
import pytorch_kinematics.transforms as tf

ROOT = pathlib.Path(__file__).resolve().parents[1]
from scipy.spatial.transform import Rotation as R

import random
from gym import spaces

class AllegroEnv:
    """
    base class for Allegro hand environment
    NOTE: in isaac gym, the orientation of object is represented as quaternion of XYZ W instead of WXYZ
    but pytorch3d and pytorch kinematics uses WXYZ convention
    and scipy transform uses XYZW convention
    """

    def __init__(self, num_envs,
                 hand_p,
                 hand_r,
                 camera_pos,
                 camera_target,
                 steps_per_action=60,
                 control_mode='joint_impedance',
                 viewer=False,
                 device='cuda:0',
                 friction_coefficient=1.0,
                 contact_controller=False,
                 video_save_path=None,
                 joint_stiffness=6.0,
                 fingers=['index', 'thumb'],  # order matters, please follow index, middle, ring, thumb
                 gravity=True,
                 gradual_control=False,
                 randomize_obj_start=False,
                 action_offset=False,
                 arm_type='None', # choose between none, 'robot', 'floating_3d'
                 ):
        if arm_type == 'robot':
            urdf = 'xela_models/victor_allegro.urdf'
        elif arm_type == 'None':
            urdf='xela_models/allegro_hand_right.urdf'
        elif arm_type == 'floating_3d':
            urdf = 'xela_models/allegro_hand_right_floating_3d.urdf'
        elif arm_type == 'floating_6d':
            urdf = 'xela_models/allegro_hand_right_floating_6d.urdf'
        else:
            raise ValueError('Invalid arm type')
        self.gym = gymapi.acquire_gym()
        self.steps_per_action = steps_per_action
        self.control_mode = control_mode
        self.num_envs = num_envs
        self.device = device
        if device == 'cpu':
            self.device_id = 0
        else:
            self.device_id = int(device.rsplit(':', 1)[-1])
        self.assets = []
        self.joint_stiffness = joint_stiffness
        # self.joint_stiffness = 50
        self.fingers = fingers
        self.num_fingers = len(fingers)
        self.friction_coefficient = friction_coefficient
        # the indexing of the joints are ranked alphabetically, rather than the order in the urdf file
        self.asset_root = f'{ROOT}/assets'
        urdf_fpath = f'{self.asset_root}/{urdf}'
        self.chain = pk.build_chain_from_urdf(open(urdf_fpath).read())

        self.robot_p = hand_p
        self.robot_r = hand_r

        self.randomize_obj_start = randomize_obj_start
        self.action_offset = action_offset

        assert control_mode in ['cartesian_impedance', 'joint_torque', 'joint_impedance', 'joint_torque_position']
        self.contact_controller = contact_controller
        if contact_controller and control_mode != 'joint_impedance':
            raise ValueError('Contact controller only works with joint impedance control')
        self.gradual_control = gradual_control
        self.arm_type = arm_type
        if arm_type == 'None':
            self.robot_dof = 4 * self.num_fingers
            self.arm_dof = 0
        elif arm_type == 'robot':
            self.robot_dof = 7 + 4 * self.num_fingers
            self.arm_dof = 7
        elif arm_type == 'floating_3d':
            self.robot_dof = 3 + 4 * self.num_fingers
            self.arm_dof = 3
        elif arm_type == 'floating_6d':
            self.robot_dof = 6 + 4 * self.num_fingers
            self.arm_dof = 6

        sim_params = gymapi.SimParams()
        sim_params.dt = 1. / 60
        sim_params.substeps = 1
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = 8
        # sim_params.physx.max_gpu_contact_pairs = int(pow(2, 23))
        if device == 'cpu':
            sim_params.physx.use_gpu = False
        else:
            sim_params.physx.use_gpu = True
            sim_params.use_gpu_pipeline = True
        if gravity:
            sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        else:
            sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
        # self.sim = self.gym.create_sim(int(self.device[-1]), 0, gymapi.SIM_PHYSX, sim_params)

        self.sim = self.gym.create_sim(self.device_id,
                                       self.device_id, gymapi.SIM_PHYSX, sim_params)

        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
        # set up the camera
        self.viewer = None
        if viewer:
            viewer_props = gymapi.CameraProperties()
            viewer_props.use_collision_geometry = True
            self.viewer = self.gym.create_viewer(self.sim, viewer_props)
            cam_pos = gymapi.Vec3(camera_pos[0], camera_pos[1], camera_pos[2])
            cam_target = gymapi.Vec3(camera_target[0], camera_target[1], camera_target[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = not gravity
        asset_options.thickness = 0.001
        asset_options.armature = 0.001
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.replace_cylinder_with_capsule = True
        self.asset_options = asset_options

        asset_options.use_mesh_materials = True
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.override_com = False
        asset_options.override_inertia = False
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params = gymapi.VhacdParams()
        asset_options.vhacd_params.resolution = 10000

        # asset_options.disable_gravity = True
        allegro_asset = self.gym.load_asset(self.sim, self.asset_root, urdf, asset_options)
        # asset_options.disable_gravity = not gravity
        # Get joint limits
        allegro_dof_props = self.gym.get_asset_dof_properties(allegro_asset)
        
        allegro_lower_limits = allegro_dof_props['lower']
        allegro_upper_limits = allegro_dof_props['upper']
        allegro_ranges = allegro_upper_limits - allegro_lower_limits
        allegro_mids = 0.5 * (allegro_upper_limits + allegro_lower_limits)
        num_dofs = len(allegro_dof_props)
        # set to effort mode
        if (control_mode == 'joint_impedance') or control_mode == 'joint_torque_position':
            allegro_dof_props['driveMode'].fill(gymapi.DOF_MODE_POS)
            allegro_dof_props['stiffness'][:self.arm_dof] = 1000 * self.joint_stiffness
            allegro_dof_props['damping'][:self.arm_dof] = 1000.0
            allegro_dof_props['stiffness'][self.arm_dof:] = self.joint_stiffness
            allegro_dof_props['damping'][self.arm_dof:] = 1.0
            # else:
            #     allegro_dof_props['stiffness'][:] = self.joint_stiffness  # zero passive stiffness
            #     allegro_dof_props['damping'][:] = 1.0
            self.kp = self.joint_stiffness * torch.eye(num_dofs)
            # self.kp_inv = torch.linalg.inv(self.kp).unsqueeze(0).to(self.device)
        else:
            allegro_dof_props['driveMode'].fill(gymapi.DOF_MODE_EFFORT)
            allegro_dof_props['stiffness'][:] = 0.0  # zero passive stiffness
            allegro_dof_props['damping'][:] = 0.0  # zero passive damping

        # set up the env grid
        spacing = 1.5
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        num_per_row = int(math.sqrt(num_envs))
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*self.robot_p)
        # NOTE: for isaac gym quat, angle goes last, but for pytorch kinematics, angle goes first 
        pose.r = gymapi.Quat(*self.robot_r)
        self.world_trans = tf.Transform3d(pos=torch.tensor(self.robot_p, device=self.device),
                                          rot=torch.tensor(
                                              [self.robot_r[3], self.robot_r[0], self.robot_r[1], self.robot_r[2]],
                                              device=self.device), device=self.device)

        self.assets.append(
            {'name': 'allegro',
             'asset': allegro_asset,
             'pose': pose,
             'dof_props': allegro_dof_props
             }
        )
        finger_to_ee_name = {
            'index': 'allegro_hand_hitosashi_finger_finger_0_aftc_base_link',
            'middle': 'allegro_hand_naka_finger_finger_1_aftc_base_link',
            'ring': 'allegro_hand_kusuri_finger_finger_2_aftc_base_link',
            'thumb': 'allegro_hand_oya_finger_3_aftc_base_link'
        }
        # NOTE: very important, the index is not the same as that in our algorithm. For isaac gym, it orders alphabetically. 
        self.finger_to_joint_index = {
            'index': [0, 1, 2, 3],
            'middle': [8, 9, 10, 11],
            'ring': [4, 5, 6, 7],
            'thumb': [12, 13, 14, 15]
        }
        for finger in self.finger_to_joint_index.keys():
            self.finger_to_joint_index[finger] = (np.array(self.finger_to_joint_index[finger]) + self.arm_dof).tolist()
        if self.arm_type == 'robot':
            self.arm_index = [0, 1, 2, 3, 4, 5, 6]
        elif self.arm_type == 'floating_3d':
            self.arm_index = [0, 1, 2]
        elif self.arm_type == 'floating_6d':
            self.arm_index = [0, 1, 2, 3, 4, 5]
        elif self.arm_type == 'None':
            self.arm_index = [] 
                        

        # self.ee_names = [finger_to_ee_name[f] for f in fingers]
        self.finger_ee_index = {finger: self.gym.find_asset_rigid_body_index(allegro_asset, finger_to_ee_name[finger])
                                for finger in self.fingers}
        self.num_dofs = num_dofs

        self._rb_states, self.rb_states = None, None
        self._actor_rb_states, self.actor_rb_states = None, None
        self._dof_states, self.dof_states = None, None
        self._q, self._qd = None, None
        self._ft_data, self.ft_data = None, None
        # self._forces, self.forces = None, None
        # self._jacobian, self.jacobian = None, None
        self.J_ee = None
        self._massmatrix, self.M = None, None
        self.default_dof_pos = None

        self.save_image_fpath = None
        self.frame_fpath = video_save_path
        self.frame_id = 0

        # self._create_env(self.assets)

    def _create_env(self, assets):
        self.envs = []
        self.handles = {}
        for asset in assets:
            self.handles[asset['name']] = []

        spacing = 1.5
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        num_per_row = int(math.sqrt(self.num_envs))

        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)
            for asset in assets:
                if asset['name'] == 'allegro':
                    allegro_asset = asset
                    assert allegro_asset['name'] == 'allegro'
                    handle = self.gym.create_actor(env, allegro_asset['asset'], allegro_asset['pose'], allegro_asset['name'], i,
                                           0)
                    self.gym.set_actor_dof_properties(env, handle, allegro_asset['dof_props'])
                    self.handles['allegro'].append(handle)
                    allegro_shape_props = self.gym.get_asset_rigid_shape_properties(allegro_asset['asset'])
                    for j in range(len(allegro_shape_props)):
                        allegro_shape_props[j].friction = self.friction_coefficient
                    self.gym.set_actor_rigid_shape_properties(self.envs[i], self.handles['allegro'][0], allegro_shape_props)
                elif asset['name'] == 'valve':
                    handle = self.gym.create_actor(env, asset['asset'], asset['pose'], asset['name'], i, 0)
                    free_dofs = [0]
                    dof_props = self.gym.get_actor_dof_properties(env, handle)
                    dof_props['driveMode'][free_dofs] = gymapi.DOF_MODE_NONE
                    dof_props['stiffness'][free_dofs] = 0
                    dof_props['damping'][free_dofs] = 0.5
                    self.gym.set_actor_dof_properties(env, handle, dof_props)
                    self.handles[asset['name']].append(handle)
                elif asset['name'] == 'screwdriver':
                    handle = self.gym.create_actor(env, asset['asset'], asset['pose'], asset['name'], i, 0)
                    # free_dofs = [0, 1, 2, 3, 4, 5, 6]
                    free_dofs = [0, 1, 2, 3]
                    dof_props = self.gym.get_actor_dof_properties(env, handle)
                    dof_props['driveMode'][free_dofs] = gymapi.DOF_MODE_NONE
                    dof_props['stiffness'][free_dofs] = 0
                    dof_props['damping'][free_dofs] = 0
                    dof_props['damping'][[0, 1]] = 0.0001
                    dof_props['damping'][2] = 0.05
                    self.gym.set_actor_dof_properties(env, handle, dof_props)
                    self.handles[asset['name']].append(handle)
                elif asset['name'] == 'cuboid' or asset['name'] == 'card' or asset['name'] == 'batarang' or asset['name'] == 'short_cuboid':
                    handle = self.gym.create_actor(env, asset['asset'], asset['pose'], asset['name'], i, 0)
                    free_dofs = [0, 1, 2, 3, 4, 5]
                    dof_props = self.gym.get_actor_dof_properties(env, handle)
                    dof_props['driveMode'][free_dofs] = gymapi.DOF_MODE_NONE
                    dof_props['stiffness'][free_dofs] = 0
                    # dof_props['damping'][free_dofs] = 0.3
                    dof_props['damping'][free_dofs] = 0.001
                    self.gym.set_actor_dof_properties(env, handle, dof_props)
                    self.handles[asset['name']].append(handle)
                elif asset['name'] == 'table' or asset['name'] == 'wall':
                    handle = self.gym.create_actor(env, asset['asset'], asset['pose'], asset['name'], i, 0)
                    dof_props = self.gym.get_actor_dof_properties(env, handle)
                    self.gym.set_actor_dof_properties(env, handle, dof_props)
                    self.handles[asset['name']].append(handle)
    def prepare_tensors(self):
        # prepare tensors for GPU usage -- must use tensor API from here on out
        self.gym.prepare_sim(self.sim)

        # state tensor
        self._rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(self._rb_states)
        self._actor_rb_states = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.actor_rb_states = gymtorch.wrap_tensor(self._actor_rb_states).view(self.num_envs, -1, 13)

        # DOF state tensor
        self._dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(self._dof_states).view(self.num_envs, -1, 2)
        self._q = self.dof_states[..., 0]
        self._qd = self.dof_states[..., 1]

        self._massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, 'allegro')

    def reset(self, env_idx=None):
        num_actors = self.actor_rb_states.shape[1]
        global_indices = torch.arange(self.num_envs * num_actors,
                                      dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        default_dof_pos = torch.zeros(self.default_dof_pos.shape).to(self.device)
        default_dof_pos[:, :self.arm_dof] = self.default_dof_pos.detach().clone()[:, :self.arm_dof]
        for i, finger in enumerate(['index', 'middle', 'ring', 'thumb']):
            idx = [self.arm_dof + i * 4, self.arm_dof + i * 4 + 1, self.arm_dof + i * 4 + 2, self.arm_dof + i * 4 + 3]
            default_dof_pos[:, self.finger_to_joint_index[finger]] = self.default_dof_pos[:, idx].detach().clone()
        default_dof_pos[:, (16 + self.arm_dof):] = self.default_dof_pos[:, (16 + self.arm_dof):].detach().clone()
        self.dof_states[:, :, 0] = default_dof_pos.detach().clone()
        self.dof_states[:, :, 1] *= 0

        if self.randomize_obj_start:
            # self.dof_states[:, 16:16+2, 0] = default_dof_pos[:, 16:16+2] + 0.05 * torch.randn_like(self.default_dof_pos[:, :16:16+2])
            self.dof_states[:, 18, 0] = default_dof_pos[:, 18] + np.pi * 2 * (torch.rand_like(self.default_dof_pos[:, 18]) - 0.5)

        if env_idx is None:
            robot_ids = global_indices[:, self.handles['allegro'][0]].contiguous()
        else:
            robot_ids = global_indices[env_idx, self.handles['allegro'][0]].contiguous()

        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_states))

        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(default_dof_pos),
                                                        gymtorch.unwrap_tensor(robot_ids),
                                                        self.num_envs
                                                        )
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(torch.zeros_like(self.default_dof_pos.clone())),
                                                        gymtorch.unwrap_tensor(robot_ids),
                                                        self.num_envs
                                                        )
        if self.viewer is not None:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)
        # to resolve contact
        for _ in range(32):
            self._step_sim()
        self._refresh_tensors()

    def set_pose(self, pose, semantic_order=True, zero_velocity=True):
        # semantic order: index, middle, ring, thumb. If the input is in this order, we have to swap the order
        # to match that in sim
        pose = pose.to(self.device)
        if len(pose.shape) == 2:
            pose = pose.unsqueeze(-1)
        if zero_velocity:
            tmp = torch.zeros_like(pose)
            pose = torch.cat((pose, tmp), dim=-1)
        if semantic_order:
            tmp = self.dof_states.clone()
            # swap the order to match that in sim
            for i, finger in enumerate(self.fingers):
                idx = [i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3]
                tmp[..., self.finger_to_joint_index[finger], :] = pose[..., idx, :]
            tmp[..., 16:, :] = pose[..., 4 * self.num_fingers:, :]
        else:
            tmp = pose
        assert pose.shape[-1] == 2
        self.dof_states[:, :, 0] = tmp[...,0]
        self.dof_states[:, :, 1] = tmp[:, :, 1]
        num_actors = self.actor_rb_states.shape[1]
        global_indices = torch.arange(self.num_envs * num_actors,
                                      dtype=torch.int32, device=self.device).view(self.num_envs, -1)

        robot_ids = global_indices[:, self.handles['allegro'][0]].contiguous()
        
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_states))

        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.default_dof_pos),
                                                        gymtorch.unwrap_tensor(robot_ids),
                                                        self.num_envs
                                                        )
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(torch.zeros_like(self.default_dof_pos)),
                                                        gymtorch.unwrap_tensor(robot_ids),
                                                        self.num_envs
                                                        )
        if self.viewer is not None:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)
        # self._step_sim()
        self._refresh_tensors()

    def get_sim(self):
        return self.sim, self.gym, self.viewer

    def _step_sim(self):
        # simulation step
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

    def _refresh_tensors(self):
        # refresh tensors
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        # self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

    def single_step(self, actions):
        des_q = None
        torques = None
        if self.control_mode == 'joint_torque':
            torques = actions

        elif self.control_mode == 'joint_impedance':
            des_q = self.default_dof_pos[:, :(self.arm_dof + 16)].clone().float()
            tmp = des_q.clone()[:, (4+self.arm_dof):(8+self.arm_dof)]
            des_q[:, (4+self.arm_dof):(8+self.arm_dof)] = des_q[:, (8+self.arm_dof):(12+self.arm_dof)]
            des_q[:, (8+self.arm_dof):(12+self.arm_dof)] = tmp
            for i, finger in enumerate(self.fingers):
                des_q[:, self.finger_to_joint_index[finger]] = actions[:, self.arm_dof + i * 4: self.arm_dof + (i + 1) * 4]
            # add palm movement
            des_q[:, :self.arm_dof] = actions[:, :self.arm_dof]
            des_q = torch.cat((des_q, self._q[:, (16+self.arm_dof):]), dim=-1)

        if torques is not None:
            torques = torch.zeros((self.num_envs, 16)).float().to(self.device)
            for i, finger in enumerate(self.fingers):
                torques[:, self.finger_to_joint_index[finger]] = actions[:, i * 4:(i + 1) * 4]
        # apply action
        if torques is not None:
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))
        else:
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(des_q))

        self._step_sim()
        self._refresh_tensors()

        # update viewer
        if self.viewer is not None:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)

    def step(self, actions, ignore_img=False):
        actions = actions.to(self.device)
        if self.action_offset:
            finger_order = {'index': 0, 'middle': 1, 'ring': 2, 'thumb': 3}
            arm_default_dof = self.default_dof_pos[:, :self.arm_dof]
            new_actions = actions.clone()
            new_actions[:, :self.arm_dof] += arm_default_dof
            for i, finger in enumerate(self.fingers):
                new_actions[:, self.arm_dof + i * 4: self.arm_dof + (i + 1) * 4] += self.default_dof_pos[:, self.arm_dof + finger_order[finger] * 4: self.arm_dof + (finger_order[finger] + 1) * 4]
            actions = new_actions
        
        if self.gradual_control:
            state = self.get_state()
            robot_q = state[:, :self.robot_dof]
        for i in range(self.steps_per_action):
            if self.gradual_control:
                if i < self.steps_per_action * 0.75:
                    temp_action = (i + 1) / (self.steps_per_action * 0.75) * (actions - robot_q) + robot_q
                else:
                    temp_action = actions
                self.single_step(temp_action)
            else:
                self.single_step(actions)
            if self.frame_fpath is not None and i % 20 == 0:
                if not ignore_img:
                    self.gym.write_viewer_image_to_file(self.viewer, f'{self.frame_fpath}/frame_{self.frame_id:06d}.png')
                    self.frame_id += 1
        return self.get_state()

    def get_state(self):
        arm_q = {'arm_q': self._q[:, self.arm_index]}
        finger_q = {finger + '_q': self._q[:, self.finger_to_joint_index[finger]] for finger in self.fingers}
        finger_ee_pos = {finger + '_pos': self.rb_states[self.finger_ee_index[finger], :3] for finger in self.fingers}
        if self.arm_type != 'None':
            results = {**finger_q, **finger_ee_pos, **arm_q}
        else:
            results = {**finger_q, **finger_ee_pos}
        return results
 
    def check_validity(self, state):
        raise NotImplementedError

class AllegroValveTurningEnv(AllegroEnv):
    """In this environment, we assume we only have access to two fingers, and we want to turn a cuboid valve"""

    def __init__(self, num_envs,
                 steps_per_action=60,
                 control_mode='cartesian_impedance',
                 viewer=False,
                 device='cuda:0',
                 friction_coefficient=1.0,
                 contact_controller=False,
                 valve_type='cylinder',
                 video_save_path=None,
                 joint_stiffness=6.0,
                 random_robot_pose=False,
                 fingers=['index', 'thumb'],  # order matters, please follow index, middle, ring, thumb,
                 gravity=True,
                 ):
        self.random_robot_pose = random_robot_pose
        cam_target = [0.0, -0.15, .40]
        if valve_type == 'cross_valve':
            cam_pos = [-0.25, 0.0, .48]
            p = np.array([0.02, -0.35, .376]).astype(np.float32)
            r = [0, 0, 0.7071068, 0.7071068]
        else:
            cam_pos = [-0.05, -0.55, .48]
            p = np.array([0.086, -0.15, .376]).astype(np.float32)
            r = [-0.0174524, 0, 0.9998477, 0]
        if random_robot_pose:
            self.random_bias = np.random.uniform(-0.02, 0.02, size=3).astype(np.float32)
            p += self.random_bias

        print("robot pose", p, r)
        super(AllegroValveTurningEnv, self).__init__(num_envs, hand_p=p, hand_r=r, camera_pos=cam_pos,
                                                     camera_target=cam_target, steps_per_action=steps_per_action,
                                                     control_mode=control_mode, viewer=viewer, device=device,
                                                     friction_coefficient=friction_coefficient,
                                                     contact_controller=contact_controller,
                                                     video_save_path=video_save_path, joint_stiffness=joint_stiffness,
                                                     fingers=fingers, gravity=gravity)

        # load valve 
        obj_pose = gymapi.Transform()
        self.obj_pose = np.array([0.0, 0.0, 0.40])
        obj_pose.p = gymapi.Vec3(*self.obj_pose)

        if valve_type == 'cylinder_valve':
            valve_urdf = 'valve/valve_cylinder.urdf'
        elif valve_type == 'cuboid_valve':
            valve_urdf = 'valve/valve_cuboid.urdf'
        elif valve_type == 'cross_valve':
            valve_urdf = 'valve/valve_cross.urdf'
        valve_asset = self.gym.load_asset(self.sim, self.asset_root, valve_urdf, self.asset_options)
        valve_shape_props = self.gym.get_asset_rigid_shape_properties(valve_asset)
        for i in range(len(valve_shape_props)):
            valve_shape_props[i].friction = friction_coefficient
        self.assets.append(

            {'name': 'valve',
             'asset': valve_asset,
             'pose': obj_pose,
             }
        )

        self._create_env(self.assets)
        self.gym.set_actor_rigid_shape_properties(self.envs[0], self.handles['valve'][0], valve_shape_props)
        self.prepare_tensors()
        # NOTE: it's in the order of index, ring, middle ,thumb
        if valve_type == 'cross_valve':
            self.default_dof_pos = torch.cat((torch.tensor([[0.3, 0.55, 0.7, 0.8]]).float(),
                                          torch.tensor([[-0.1, 0.2, 0.9, 0.8]]).float(),
                                          torch.tensor([[0.0, 0.0, 0.0, 0.0]]).float(),
                                          torch.tensor([[1.0, 0.1, 0.3, 1.0]]).float()),
                                          dim=1).to(self.device)
        else:
            self.default_dof_pos = torch.cat((torch.tensor([[0., 0.5, 0.7, 0.7]]).float(),
                                          torch.tensor([[0., 0.5, 0.7, 0.7]]).float(),
                                          torch.tensor([[0., 0.5, 0.7, 0.7]]).float(),
                                          torch.tensor([[1.3, 0.1, -0.1, 1.0]]).float()),
                                          dim=1).to(self.device)
        # add the valve angle to it
        self.default_dof_pos = torch.cat((self.default_dof_pos, torch.zeros((1, 1)).float().to(device=self.device)),
                                         dim=1)
        self.default_dof_pos = self.default_dof_pos.repeat(self.num_envs, 1)
        self.reset()


    def get_state(self):
        results = super(AllegroValveTurningEnv, self).get_state()
        results['valve'] = self._q[:, -1:]
        q = []
        for finger in self.fingers:
            q.append(results[f'{finger}_q'])
        q.append(results['valve'])
        q = torch.cat(q, dim=1)
        results['q'] = q
        return q

    def get_valve_inertia(self):
        valve_handle = self.handles['valve'][0]
        properties = self.gym.get_actor_rigid_body_properties(self.envs[0], valve_handle)
        inertia = properties[1].inertia  # Mat33 object
        return inertia
    def check_validity(self, state):
        return True


class AllegroScrewdriverTurningEnv(AllegroEnv):
    def __init__(self, num_envs,
                 steps_per_action=60,
                 control_mode='cartesian_impedance',
                 viewer=False,
                 device='cuda:0',
                 friction_coefficient=1.0,
                 contact_controller=False,
                 video_save_path=None,
                 joint_stiffness=6.0,
                 fingers=['index', 'thumb'],  # order matters, please follow index, middle, ring, thumb
                 obj_pose=None,
                 gradual_control=False,
                 gravity=True,
                 randomize_obj_start=False,
                 arm_type='None',
                 ):
        cam_pos = [-0.15, 0.2, 1.49]
        cam_target = [0.0, 0.0, 1.335]
        if arm_type == 'robot':
            p = [-0.8, 0, 0]
            r = [0, 0, 0, 1]
        elif arm_type == 'None' or arm_type == "floating_3d" or arm_type == "floating_6d":
            p = [0, -0.095, 1.33]
            r = [0.2418448, 0.2418448, 0.664463, 0.664463]# 40 degrees
        super(AllegroScrewdriverTurningEnv, self).__init__(num_envs, hand_p=p, hand_r=r, camera_pos=cam_pos,
                                                           camera_target=cam_target, steps_per_action=steps_per_action,
                                                           control_mode=control_mode, viewer=viewer, device=device,
                                                           friction_coefficient=friction_coefficient,
                                                           contact_controller=contact_controller,
                                                           video_save_path=video_save_path,
                                                           joint_stiffness=joint_stiffness, fingers=fingers,
                                                           gradual_control=gradual_control,
                                                           gravity=gravity,
                                                           randomize_obj_start=randomize_obj_start,
                                                           arm_type=arm_type) 
        obj_pose_tf = gymapi.Transform()
        if obj_pose is None:
            self.obj_pose = np.array([0, 0, 1.205])
            obj_pose_tf.p = gymapi.Vec3(*self.obj_pose)
        else:
            obj_pose_tf.p = gymapi.Vec3(*obj_pose)
            self.obj_pose = np.array(obj_pose)

        screwdriver_urdf = 'screwdriver/screwdriver.urdf'
        obj_urdf_fpath = f'{self.asset_root}/{screwdriver_urdf}'
        self.object_chain = pk.build_chain_from_urdf(open(obj_urdf_fpath, 'r').read())
        self.nominal_screwdriver_top = np.array([0, 0, 1.405]) # in the world frame for validity checking

        self.asset_options.replace_cylinder_with_capsule = False
        screwdriver_asset = self.gym.load_asset(self.sim, self.asset_root, screwdriver_urdf, self.asset_options)
        screwdriver_shape_props = self.gym.get_asset_rigid_shape_properties(screwdriver_asset)
        for i in range(len(screwdriver_shape_props)):
            screwdriver_shape_props[i].friction = friction_coefficient
        self.assets.append(
            {'name': 'screwdriver',
             'asset': screwdriver_asset,
             'pose': obj_pose_tf,
             }
        )

        self._create_env(self.assets)
        self.gym.set_actor_rigid_shape_properties(self.envs[0], self.handles['screwdriver'][0], screwdriver_shape_props)
        self.prepare_tensors()

        # requires pregrasp
        self.default_dof_pos = torch.cat((
                                    torch.tensor([[0.1, 0.6, 0.6, 0.6]]).float(),
                                    torch.tensor([[-0.1, 0.5, 0.9, 0.9]]).float(),
                                    torch.tensor([[0., 0.5, 0.65, 0.65]]).float(),
                                    torch.tensor([[1.2, 0.3, 0.3, 1.2]]).float()
                                    ),
                                    dim=1).to(self.device)

        if self.arm_type != 'None':
            if self.arm_type == 'robot':
                self.arm_default_dof = torch.tensor([[-0.4627,  0.5445,  0.3865, -1.6972, -1.1118, -1.4570,  0.1162]]).to(device=self.device)
            elif self.arm_type == 'floating_3d':
                self.arm_default_dof = torch.tensor([[0.0, 0.0, 0.0]]).to(device=self.device)
            elif self.arm_type == 'floating_6d':
                self.arm_default_dof = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).to(device=self.device)
            self.default_dof_pos = torch.cat((self.arm_default_dof, self.default_dof_pos), dim=1)

        # add the screwdriver angle to it
        self.default_dof_pos = torch.cat((self.default_dof_pos, torch.zeros((1, 4)).float().to(device=self.device)),
                                         dim=1).to(self.device)
        self.default_dof_pos = self.default_dof_pos.repeat(self.num_envs, 1)
        self.reset()


    def get_state(self):
        results = super(AllegroScrewdriverTurningEnv, self).get_state()
        screwdriver_ori_euler = self._q[:, -4:-1]
        screwdriver_ori_axis_angle = R.from_euler('XYZ', screwdriver_ori_euler.cpu().numpy()).as_rotvec()
        screwdriver_ori_axis_angle = torch.tensor(screwdriver_ori_axis_angle).to(device=self.device).float()

        results['screwdriver_ori_euler'] = screwdriver_ori_euler
        results['screwdriver_ori_axis_angle'] = screwdriver_ori_axis_angle
        # results['screwdriver_ori'] = screwdriver_ori_euler  # keeps using the euler angle since the pytorch volumetric might have to use it.
        results['screwdriver_ori'] = screwdriver_ori_axis_angle  # keeps using the euler angle since the pytorch volumetric might have to use it.
        results['screwdriver_angle'] = self._q[:, -1:]

        results['screwdriver_ori_euler'] = screwdriver_ori_euler
        results['screwdriver_ori_axis_angle'] = screwdriver_ori_axis_angle
        results['screwdriver_ori'] = screwdriver_ori_euler  # keeps using the euler angle since the pytorch volumetric might have to use it.
        results['screwdriver_angle'] = self._q[:, -1:]
        q = []
        if self.arm_type != 'None':
            q.append(results['arm_q'])
        for finger in self.fingers:
            q.append(results[f'{finger}_q'])
        q.append(results['screwdriver_ori'])
        # q.append(results['screwdriver_angle'])
        q = torch.cat(q, dim=1)
        results['q'] = q
        return q
    def check_validity(self, state):
        if state.dim() == 2:
            state = state[0]
        obj_dof = 3
        screwdriver_state = state[-obj_dof:]
        screwdriver_top_pos = self.get_screwdriver_top_in_world(screwdriver_state)
        screwdriver_top_pos = screwdriver_top_pos.detach().cpu().numpy()
        distance2nominal = np.linalg.norm(screwdriver_top_pos - self.nominal_screwdriver_top)
        if distance2nominal > 0.02:
            validity_flag = False
        else:
            validity_flag = True
        return validity_flag
    def get_screwdriver_top_in_world(self, env_q):
        """
        env_q: 1 dimension without batch
        """
        env_q = torch.cat((env_q, torch.zeros(1, device=env_q.device)), dim=-1) # add the screwdriver cap dim
        screwdriver_top_obj_frame = self.object_chain.forward_kinematics(env_q.unsqueeze(0).to(self.object_chain.device))['screwdriver_cap']
        screwdriver_top_obj_frame = screwdriver_top_obj_frame.get_matrix().reshape(4, 4)[:3, 3]
        world2obj_trans = tf.Transform3d(pos=torch.tensor(self.obj_pose, device=self.object_chain.device).float(),
                                            rot=torch.tensor([1, 0, 0, 0], device=self.object_chain.device).float(), device=self.object_chain.device)
        screwdriver_top_world_frame = world2obj_trans.transform_points(screwdriver_top_obj_frame.unsqueeze(0)).squeeze(0)
        return screwdriver_top_world_frame


class AllegroScrewdriverEnv(AllegroEnv):
    "6D screwdriver environment"

    def __init__(self, num_envs,
                 steps_per_action=60,
                 control_mode='cartesian_impedance',
                 viewer=False,
                 device='cuda:0',
                 friction_coefficient=1.0,
                 contact_controller=False,
                 video_save_path=None,
                 joint_stiffness=6.0,
                 fingers=['index', 'thumb'],  # order matters, please follow index, middle, ring, thumb
                 gravity=True,
                 gradual_control=False,
                 ):
        cam_pos = [-0.3, 0.4, 0.38]
        cam_target = [0.0, 0.0, 0.305]
        p = [0.01, -0.028, 0.31]
        r = [-0.5, 0.5, 0.5, 0.5]
        super(AllegroScrewdriverEnv, self).__init__(num_envs, hand_p=p, hand_r=r, camera_pos=cam_pos,
                                                    camera_target=cam_target, steps_per_action=steps_per_action,
                                                    control_mode=control_mode, viewer=viewer, device=device,
                                                    friction_coefficient=friction_coefficient,
                                                    contact_controller=contact_controller,
                                                    video_save_path=video_save_path, joint_stiffness=joint_stiffness,
                                                    fingers=fingers, gravity=gravity, gradual_control=gradual_control)
        obj_pose = gymapi.Transform()
        self.obj_pose = np.array([0, 0, 0.205])
        obj_pose.p = gymapi.Vec3(*self.obj_pose)
        

        screwdriver_urdf = 'screwdriver/screwdriver_6d.urdf'
        self.asset_options.replace_cylinder_with_capsule = False
        screwdriver_asset = self.gym.load_asset(self.sim, self.asset_root, screwdriver_urdf, self.asset_options)
        screwdriver_shape_props = self.gym.get_asset_rigid_shape_properties(screwdriver_asset)
        for i in range(len(screwdriver_shape_props)):
            screwdriver_shape_props[i].friction = friction_coefficient
        self.assets.append(
            {'name': 'screwdriver',
             'asset': screwdriver_asset,
             'pose': obj_pose,
             }
        )

        self._create_env(self.assets)
        self.gym.set_actor_rigid_shape_properties(self.envs[0], self.handles['screwdriver'][0], screwdriver_shape_props)
        self.prepare_tensors()

        self.default_dof_pos = torch.cat((torch.tensor([[0., 0.5, 0.7, 0.7]]).float().to(device=self.device),
                                          torch.tensor([[0., 0.5, 0.7, 0.7]]).float().to(device=self.device),
                                          torch.tensor([[0., 0.5, 0.7, 0.7]]).float().to(device=self.device),
                                          torch.tensor([[1.3, 0.3, 0.2, 1.1]]).float().to(device=self.device)),
                                         dim=1).to(self.device)
        # add the screwdriver angle to it
        screwdriver_default_pos = torch.tensor([0, 0, 0, 0, -1.57, 0, 0]).float().to(device=self.device)
        self.default_dof_pos = torch.cat((self.default_dof_pos, screwdriver_default_pos.unsqueeze(0)),
                                         dim=1).to(self.device)
        self.default_dof_pos = self.default_dof_pos.repeat(self.num_envs, 1)
        self.reset()

    def get_state(self):
        results = super(AllegroScrewdriverEnv, self).get_state()
        screwdriver_ori_euler = self._q[:, -4:-1]
        screwdriver_position = self._q[:, -7:-4]
        results['screwdriver_ori_euler'] = screwdriver_ori_euler
        results['screwdriver_ori'] = screwdriver_ori_euler
        results['screwdriver_position'] = screwdriver_position
        results['screwdriver_angle'] = self._q[:, -1:]
        # gt_euler = R.from_quat(self.rb_states[-4, 3:7].cpu()).as_euler('XYZ')
        # print(gt_euler, screwdriver_ori_euler)
        q = []
        for finger in self.fingers:
            q.append(results[f'{finger}_q'])
        q.append(results['screwdriver_position'])
        q.append(results['screwdriver_ori'])
        q.append(results['screwdriver_angle'])
        q = torch.cat(q, dim=1)
        results['q'] = q
        return results

class AllegroCuboidAlignmentEnv(AllegroEnv):
    def __init__(self, num_envs,
                 steps_per_action=60,
                 control_mode='cartesian_impedance',
                 viewer=False,
                 device='cuda:0',
                 friction_coefficient=1.0,
                 contact_controller=False,
                 video_save_path=None,
                 joint_stiffness=6.0,
                 fingers=['index', 'thumb'],  # order matters, please follow index, middle, ring, thumb
                 gravity=True,
                 gradual_control=False,
                 ):
        cam_pos = [-0.4, 0.4, 0.48]
        cam_target = [0.0, 0.0, 0.305]
        p = [0.11, -0.023, 0.30]
        r = [ 0, 0.4226183, 0.9063078, 0 ]
        super(AllegroCuboidAlignmentEnv, self).__init__(num_envs, hand_p=p, hand_r=r, camera_pos=cam_pos,
                                                     camera_target=cam_target, steps_per_action=steps_per_action,
                                                     control_mode=control_mode, viewer=viewer, device=device,
                                                     friction_coefficient=friction_coefficient,
                                                     contact_controller=contact_controller,
                                                     video_save_path=video_save_path, joint_stiffness=joint_stiffness,
                                                     fingers=fingers, gravity=gravity, gradual_control=gradual_control)
        obj_pose = gymapi.Transform()
        self.obj_pose = np.array([0.05, 0, 0.205])
        obj_pose.p = gymapi.Vec3(*self.obj_pose)

        cuboid_urdf = 'cuboid_insertion/cuboid_with_wall.urdf'
        self.asset_options.replace_cylinder_with_capsule = True
        cuboid_asset = self.gym.load_asset(self.sim, self.asset_root, cuboid_urdf, self.asset_options)
        cuboid_shape_props = self.gym.get_asset_rigid_shape_properties(cuboid_asset)
        for i in range(len(cuboid_shape_props)):
            cuboid_shape_props[i].friction = friction_coefficient
        self.assets.append(
            {'name': 'cuboid',
             'asset': cuboid_asset,
             'pose': obj_pose,
             }
        )

        # # create wall
        self.wall_dims = np.array([0.1, 0.5, 0.12])
        wall_dims = gymapi.Vec3(*self.wall_dims)
        wall_pose = gymapi.Transform()
        self.wall_pose = np.array([0, -0.25, 0.19])
        wall_pose.p = gymapi.Vec3(self.wall_pose[0], self.wall_pose[1], self.wall_pose[2])

        self._create_env(self.assets)
        self.gym.set_actor_rigid_shape_properties(self.envs[0], self.handles['cuboid'][0], cuboid_shape_props)
        self.prepare_tensors()

        self.default_dof_pos = torch.cat((torch.tensor([[0, 0.7, 0.8, 0.8]]).float(),
                                    torch.tensor([[0, 0.8, 0.7, 0.6]]).float(),
                                    torch.tensor([[0, 0.3, 0.3, 0.6]]).float(),
                                    torch.tensor([[1.2, 0.3, 0.05, 1.1]]).float()),
                                    dim=1).to(self.device)
        # add the screwdriver angle to it
        self.default_dof_pos = torch.cat((self.default_dof_pos, torch.tensor([[-0.05, 0, 0.08, 0.67, 0, 0]]).float().to(device=self.device)),
                                         dim=1)
        self.default_dof_pos = self.default_dof_pos.repeat(self.num_envs, 1)
        self.reset()

    def get_state(self):
        results = super(AllegroCuboidAlignmentEnv, self).get_state()
        cuboid_ori_euler = self._q[:, -3:].clone()
        cuboid_position = self._q[:, -6:-3].clone()
        results['cuboid_ori'] = cuboid_ori_euler
        results['cuboid_position'] = cuboid_position
        q = []
        for finger in self.fingers:
            q.append(results[f'{finger}_q'])
        q.append(results['cuboid_position'])
        q.append(results['cuboid_ori'])
        q = torch.cat(q, dim=1)
        return q
    
    def check_validity(self, state):
        obj_dof = 6
        if state.dim() == 2:
            state = state[0]
        obj_state = state[-obj_dof:]
        validity_flag = True
        if obj_state[2] < 0.0:
            validity_flag = False
        return validity_flag

class AllegroCuboidTurningEnv(AllegroEnv):
    def __init__(self, num_envs,
                 steps_per_action=60,
                 control_mode='cartesian_impedance',
                 viewer=False,
                 device='cuda:0',
                 friction_coefficient=1.0,
                 contact_controller=False,
                 video_save_path=None,
                 joint_stiffness=6.0,
                 fingers=['index', 'thumb'],  # order matters, please follow index, middle, ring, thumb
                 gravity=True,
                 gradual_control=False,
                 ):
        cam_pos = [0.3, 0.3, 0.48]
        cam_target = [0.0, 0.0, 0.305]
        p = [-0.1, -0.025, 0.30]
        r = [0.258819, 0, 0, 0.9659258]
        super(AllegroCuboidTurningEnv, self).__init__(num_envs, hand_p=p, hand_r=r, camera_pos=cam_pos,
                                                     camera_target=cam_target, steps_per_action=steps_per_action,
                                                     control_mode=control_mode, viewer=viewer, device=device,
                                                     friction_coefficient=friction_coefficient,
                                                     contact_controller=contact_controller,
                                                     video_save_path=video_save_path, joint_stiffness=joint_stiffness,
                                                     fingers=fingers, gravity=gravity,
                                                     gradual_control=gradual_control)
        obj_pose = gymapi.Transform()
        # self.obj_pose = np.array([0, 0, 0.29])
        self.obj_pose = np.array([0, 0, 0.31])
        obj_pose.p = gymapi.Vec3(*self.obj_pose)

        cuboid_urdf = 'cuboid_insertion/short_cuboid.urdf'
        self.asset_options.replace_cylinder_with_capsule = True
        cuboid_asset = self.gym.load_asset(self.sim, self.asset_root, cuboid_urdf, self.asset_options)
        cuboid_shape_props = self.gym.get_asset_rigid_shape_properties(cuboid_asset)
        for i in range(len(cuboid_shape_props)):
            cuboid_shape_props[i].friction = friction_coefficient
        self.assets.append(
            {'name': 'cuboid',
             'asset': cuboid_asset,
             'pose': obj_pose,
             }
        )


        self._create_env(self.assets)
        self.gym.set_actor_rigid_shape_properties(self.envs[0], self.handles['cuboid'][0], cuboid_shape_props)

        self.prepare_tensors()
        self.default_dof_pos = torch.cat((torch.tensor([[0.0, 0.8, 0.4, 0.7]]).float(),
                                        torch.tensor([[-0.15, 0.9, 1.0, 0.9]]).float(),
                                        torch.tensor([[0, 0.3, 0.3, 0.6]]).float(),
                                        torch.tensor([[0.7, 1.0, 0.6, 1.05]]).float()),
                                        dim=1).to(self.device)

        # add the screwdriver angle to it
        self.default_dof_pos = torch.cat((self.default_dof_pos, torch.tensor([[0, 0, 0, 0, -0.523599, 0]]).float().to(device=self.device)),
                                         dim=1)
        self.default_dof_pos = self.default_dof_pos.repeat(self.num_envs, 1)
        self.reset()

    def get_state(self):
        results = super(AllegroCuboidTurningEnv, self).get_state()
        cuboid_ori_euler = self._q[:, -3:]
        cuboid_position = self._q[:, -6:-3]
        results['cuboid_ori'] = cuboid_ori_euler
        results['cuboid_position'] = cuboid_position
        q = []
        for finger in self.fingers:
            q.append(results[f'{finger}_q'])
        q.append(results['cuboid_position'])
        q.append(results['cuboid_ori'])
        q = torch.cat(q, dim=1)
        results['q'] = q
        return q
    def check_validity(self, state):
        obj_dof = 6
        if state.dim() == 2:
            state = state[0]
        obj_state = state[-obj_dof:]
        obj_z = obj_state[2]
        if obj_z < -0.1:
            return False
        else:
            return True

class AllegroReorientationEnv(AllegroEnv):
    def __init__(self, num_envs,
                 steps_per_action=60,
                 control_mode='cartesian_impedance',
                 viewer=False,
                 device='cuda:0',
                 friction_coefficient=1.0,
                 contact_controller=False,
                 video_save_path=None,
                 joint_stiffness=6.0,
                 fingers=['index', 'thumb'],  # order matters, please follow index, middle, ring, thumb
                 gravity=True,
                 gradual_control=False,
                 ):
        cam_pos = [0., 0.3, 0.45]
        cam_target = [0.0, 0.0, 0.305]
        p = [-0.04, -0.023, 0.30]
        r = [0, 0.7071068, 0, 0.7071068 ]
        super(AllegroReorientationEnv, self).__init__(num_envs, hand_p=p, hand_r=r, camera_pos=cam_pos,
                                                     camera_target=cam_target, steps_per_action=steps_per_action,
                                                     control_mode=control_mode, viewer=viewer, device=device,
                                                     friction_coefficient=friction_coefficient,
                                                     contact_controller=contact_controller,
                                                     video_save_path=video_save_path, joint_stiffness=joint_stiffness,
                                                     fingers=fingers, gravity=gravity,
                                                     gradual_control=gradual_control)
        obj_pose = gymapi.Transform()
        self.obj_pose = np.array([0, 0, 0.21])
        obj_pose.p = gymapi.Vec3(*self.obj_pose)

        obj_urdf = 'reorientation/batarang.urdf'
        self.asset_options.replace_cylinder_with_capsule = False
        obj_asset = self.gym.load_asset(self.sim, self.asset_root, obj_urdf, self.asset_options)
        obj_shape_props = self.gym.get_asset_rigid_shape_properties(obj_asset)
        for i in range(len(obj_shape_props)):
            obj_shape_props[i].friction = friction_coefficient
        self.assets.append(
            {'name': 'batarang',
             'asset': obj_asset,
             'pose': obj_pose,
             }
        )


        self._create_env(self.assets)
        self.gym.set_actor_rigid_shape_properties(self.envs[0], self.handles['batarang'][0], obj_shape_props)

        self.prepare_tensors()

        # for flat one
        self.default_dof_pos = torch.cat((torch.tensor([[0.15, 0.7, 0.4, 0.74]]).float(),
                                        torch.tensor([[-0.15, 0.7, 0.45, 0.76]]).float(),
                                        torch.tensor([[0, 0.0, 0.0, 0.0]]).float(),
                                        torch.tensor([[1.5, 0.0, 0.25, 1.12]]).float()),
                                        dim=1).to(self.device)
                                        
        # add the screwdriver angle to it
        self.default_dof_pos = torch.cat((self.default_dof_pos, torch.tensor([[0, 0, 0, 0, 0, 0]]).float().to(device=self.device)),
                                         dim=1)
        self.default_dof_pos = self.default_dof_pos.repeat(self.num_envs, 1)
        self.reset()

    def get_state(self):
        results = super(AllegroReorientationEnv, self).get_state()
        obj_ori_euler = self._q[:, -3:]
        obj_position = self._q[:, -6:-3]
        results['obj_ori'] = obj_ori_euler
        results['obj_position'] = obj_position
        q = []
        for finger in self.fingers:
            q.append(results[f'{finger}_q'])
        q.append(results['obj_position'])
        q.append(results['obj_ori'])
        q = torch.cat(q, dim=1)
        results['q'] = q
        return q
    def check_validity(self, state):
        obj_dof = 6
        if state.dim() == 2:
            state = state[0]
        obj_state = state[-obj_dof:]
        obj_z = obj_state[2]
        if obj_z < -0.1:
            return False
        else:
            return True
    

