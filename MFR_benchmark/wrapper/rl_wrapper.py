
import torch
from gym import spaces
import numpy as np
import pytorch_kinematics.transforms as tf
from scipy.spatial.transform import Rotation as R


class RLWrapper:
    def __init__(self, env, max_episode_length, action_offset, goal, n_obs):
        self.env = env
        self.env.action_offset = action_offset
        self.max_episode_length = max_episode_length
        self.progress_buf = torch.zeros(self.env.num_envs).cuda()
        self.timeout_buf = torch.zeros(self.env.num_envs).cuda()
        self.goal = goal
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(self.env.robot_dof,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32)
        self.device = self.env.device
    def reward(self, state, action):
        raise NotImplementedError
    
    def check_done(self, state):
        return torch.zeros(self.env.num_envs).bool().cuda()
        # raise NotImplementedError
    def step(self, action):
        action = action.to(device=self.device)
        state = self.env.step(action)
        reward = self.reward(state, action)
        done = self.check_done(state)
        self.timeout_buf = torch.where(self.progress_buf >= self.max_episode_length, torch.ones_like(self.timeout_buf),
                                       torch.zeros_like(self.timeout_buf)).cuda()
        
        done = done.squeeze(-1)
        reset_buf = done | self.timeout_buf.bool()
        distance2goal = None
        if reset_buf.any():
            distance2goal = self.get_distance2goal()
            self.reset(reset_buf.cpu())
        done = reset_buf
        # dropped = (abs(state[:, -3]) > 1.0) | (abs(state[:, -2]) > 1.0)
        info = {'time_outs': self.timeout_buf, 'final_distance2goal': distance2goal}
        state = {'obs': state.cuda(), 'priv_info': state.cuda()}

        self.progress_buf += 1

        return state, reward, done, info
    def reset(self, env_idx=None): 
        self.env.reset(env_idx)
        state = self.env.get_state()
        self.progress_buf = torch.zeros(self.env.num_envs).cuda()
        self.timeout_buf = torch.zeros(self.env.num_envs).cuda()
        state = {'obs': state.cuda(), 'priv_info': state.cuda()}
        return state
    
class AllegroScrewdriverRLWrapper(RLWrapper):
    def __init__(self, env, goal=torch.tensor([[0, 0, -1.5707]]), action_offset=True, n_obs=15, max_episode_length=12):
        super().__init__(env, max_episode_length, action_offset, goal, n_obs)
        self.goal_mat = R.from_euler('XYZ', self.goal.numpy()).as_matrix()
        self.obj_dof = 3
    def reward(self, state, action):
        assert len(action.shape) == 2
        action_cost = torch.sum(action ** 2, dim=-1)

        obj_orientation = state[:, -4:-1]
        goal_cost = torch.sum((20 * (obj_orientation - self.goal.to(self.device)) ** 2), dim=-1)

        #upright cost
        upright_cost = 10000 * torch.sum(obj_orientation[:, :-1] ** 2, dim=-1)
        # dropping cost
        cost = action_cost + goal_cost + upright_cost
        cost = torch.nan_to_num(cost, nan=1e6)
        reward = -cost
        return reward.cuda()
    
    

    def get_distance2goal(self):
        state = self.env.get_state()
        screwdriver_state = state[:, -self.obj_dof:]
        screwdriver_mat = R.from_euler('XYZ', screwdriver_state.cpu()).as_matrix()
        distance2goal = tf.so3_relative_angle(torch.tensor(screwdriver_mat), \
            torch.tensor(self.goal_mat).repeat(self.env.num_envs, 1, 1), cos_angle=False).detach().cpu().abs()
        return distance2goal
        
class AllegroValveTurningRLWrapper(RLWrapper):
    def __init__(self, env, goal=torch.tensor([[-0.785398]]), action_offset=True, n_obs=13, max_episode_length=8):
        super().__init__(env, max_episode_length, action_offset, goal, n_obs)
        self.goal = goal.to(self.device)
        self.obj_dof = 1

    def reward(self, state, action):
        assert len(action.shape) == 2
        reward = 0
        # goal cost
        distance2goal = self.get_distance2goal()
        reward += -100 * torch.pow(distance2goal, 2).squeeze(-1).to(self.device)
        # action_cost
        reward -= 50.0 * (torch.norm(action, dim=-1) ** 2)

        return reward.cuda()
    def get_distance2goal(self):
        state = self.env.get_state()
        obj_state = state[:, -self.obj_dof:]
        distance2goal = obj_state - self.goal
        return distance2goal
    
class AllegrocuboidTurningRLWrapper(RLWrapper):
    def __init__(self, env, max_episode_length=10, action_offset=True, goal=torch.tensor([[0, 0, 0, 0, -1.5707, 0]]), n_obs=18):
        super().__init__(env, max_episode_length, action_offset, goal, n_obs)
        self.obj_dof = 6
        self.goal = goal.to(self.device)
        self.goal_mat =  R.from_euler('XYZ', goal[0, -3:].cpu().numpy()).as_matrix()
    def reward(self, state, action):
        assert len(action.shape) == 2
        reward = 0
        # goal cost
        obj_state = state[:, -self.obj_dof:]
        obj_pos = obj_state[:, :3]
        obj_ori = obj_state[:, 3:]

        cuboid_upright_cost = (obj_ori[:, 0] ** 2) + (obj_ori[:, 2] ** 2)
        reward -= 1000 * cuboid_upright_cost

        reward -= 10.0 * torch.sum((obj_pos - self.goal[:, :3])**2, dim=-1).to(self.device)
        distance2goal = self.get_distance2goal()
        reward -= 10 * torch.pow(distance2goal, 2).to(self.device)

        # dropp_flag = state[:, -4] < -0.1
        # reward -= 1000 * dropp_flag.to(self.device)
        # dropping cost
        reward -= 1e6 * ((state[:, -4] < -0.07) * state[:, -4])** 2

        # action_cost
        reward -= 50.0 * (torch.norm(action, dim=-1) ** 2)

        return reward.cuda()
    def get_distance2goal(self):
        state = self.env.get_state()
        obj_state = state[:, -self.obj_dof:]
        obj_pos = obj_state[:, :3]
        obj_ori = obj_state[:, 3:]
        obj_mat = R.from_euler('XYZ', obj_ori.cpu()).as_matrix()
        distance2goal = tf.so3_relative_angle(torch.tensor(obj_mat), \
            torch.tensor(self.goal_mat).repeat(self.env.num_envs, 1, 1), cos_angle=False).detach().cpu().abs()
        return distance2goal

class AllegrocuboidAlignmentRLWrapper(RLWrapper):
    def __init__(self, env, max_episode_length=10, action_offset=True, goal=torch.tensor([[0, 0, 0, 0, 0, 0]]), n_obs=18):
        super().__init__(env, max_episode_length, action_offset, goal, n_obs)
        self.obj_dof = 6
        self.goal = goal.to(self.device)
        self.goal_mat =  R.from_euler('XYZ', goal[0, -3:].cpu().numpy()).as_matrix()
    def reward(self, state, action):
        assert len(action.shape) == 2
        reward = 0
        # goal cost
        obj_state = state[:, -self.obj_dof:]
        obj_pos = obj_state[:, :3]
        obj_ori = obj_state[:, 3:]

        cuboid_upright_cost = (obj_ori[:, 1] ** 2) + (obj_ori[:, 2] ** 2)
        reward -= 1000 * cuboid_upright_cost

        reward -= 10.0 * torch.sum((obj_pos - self.goal[:, :3])**2, dim=-1).to(self.device)
        # obj_mat = R.from_euler('XYZ', obj_ori.cpu()).as_matrix()
        distance2goal = self.get_distance2goal()
        reward -= 10 * torch.pow(distance2goal, 2).to(self.device)

        # drop_flag = state[:, -4] < 0.0
        # reward -= 1000 * drop_flag.to(self.device)

        reward -= 1e6 * ((state[:, -4] < -0.02) * state[:, -4])** 2

        reward -= 50.0 * (torch.norm(action, dim=-1) ** 2)

        return reward.cuda()
    def get_distance2goal(self):
        state = self.env.get_state()
        obj_state = state[:, -self.obj_dof:]
        obj_pos = obj_state[:, :3]
        obj_ori = obj_state[:, 3:]
        obj_mat = R.from_euler('XYZ', obj_ori.cpu()).as_matrix()
        distance2goal = tf.so3_relative_angle(torch.tensor(obj_mat), \
            torch.tensor(self.goal_mat).repeat(self.env.num_envs, 1, 1), cos_angle=False).detach().cpu().abs()
        return distance2goal

    
class AllegroReorientationRLWrapper(RLWrapper):
    def __init__(self, env, max_episode_length=10, action_offset=True, goal=torch.tensor([[-0.01, 0, 0, 0, 0, -1.0472]]), n_obs=18):
        super().__init__(env, max_episode_length, action_offset, goal, n_obs)
        self.obj_dof = 6
        self.goal = goal.to(self.device)
        self.goal_mat =  R.from_euler('XYZ', goal[0, -3:].cpu().numpy()).as_matrix()
    def reward(self, state, action):
        assert len(action.shape) == 2
        reward = 0
        # goal cost
        obj_state = state[:, -self.obj_dof:]
        obj_pos = obj_state[:, :3]
        obj_ori = obj_state[:, 3:]

        cuboid_upright_cost = (obj_ori[:, 0] ** 2) + (obj_ori[:, 1] ** 2)
        reward -= 1000 * cuboid_upright_cost

        reward -= 10.0 * torch.sum((obj_pos - self.goal[:, :3])**2, dim=-1).to(self.device)
        distance2goal = self.get_distance2goal()
        reward -= 10 * torch.pow(distance2goal, 2).to(self.device)

        # drop_flag = state[:, -4] < -0.1
        # reward -= 500 * drop_flag.to(self.device)

        reward -= 1e6 * ((state[:, -4] < -0.02) * state[:, -4])** 2

        reward -= 50.0 * (torch.norm(action, dim=-1) ** 2)

        return reward.cuda()
    
    def get_distance2goal(self):
        state = self.env.get_state()
        obj_state = state[:, -self.obj_dof:]
        obj_pos = obj_state[:, :3]
        obj_ori = obj_state[:, 3:]
        obj_mat = R.from_euler('XYZ', obj_ori.cpu()).as_matrix()
        distance2goal = tf.so3_relative_angle(torch.tensor(obj_mat), \
            torch.tensor(self.goal_mat).repeat(self.env.num_envs, 1, 1), cos_angle=False).detach().cpu().abs()
        return distance2goal
