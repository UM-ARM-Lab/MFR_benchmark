from MFR_benchmark.tasks.allegro import *
from MFR_benchmark.wrapper.rl_wrapper import *
import torch
import pytorch_kinematics as pk
import pytorch_kinematics.transforms as tf
from isaacgym import gymtorch
from isaacgym import gymapi
import time
import numpy as np



if __name__ == "__main__":
    num_envs = 2
    env = AllegroScrewdriverTurningEnv(num_envs, control_mode='joint_impedance', viewer=True,
    steps_per_action=60,
    device = 'cuda:0',
    friction_coefficient=1.0,
    fingers=['index', 'middle', 'thumb'],
    arm_type='None',
    gravity=True,
    gradual_control=True,
    )
    # using an RL wrapper is optional
    env = AllegroScrewdriverRLWrapper(env)

    env.reset()
    while True:
        action = torch.randn((num_envs,12)).to(env.device) * 0
        next_state, reward, done, info = env.step(action)
        if done[0]:
            print(info['final_distance2goal'])
