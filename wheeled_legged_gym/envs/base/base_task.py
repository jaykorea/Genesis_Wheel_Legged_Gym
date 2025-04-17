import sys
import numpy as np
import torch
import time
import genesis as gs

# Base class for RL tasks
class BaseTask():

    def __init__(self, cfg, sim_device, headless):
        
        self.render_fps = 50
        self.last_frame_time = 0

        if not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            assert sim_device in ["cpu", "cuda"]
            self.device = torch.device(sim_device)
        self.headless = headless

        self.num_envs = 1 if cfg.env.num_envs == 0 else cfg.env.num_envs
        self.num_build_envs = self.num_envs
        self.num_obs = cfg.env.num_observations
        self.num_stacks = 1 if cfg.env.num_stacks == 0 else cfg.env.num_stacks
        self.num_commands = cfg.commands.num_commands
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions

        # allocate buffers
        self.single_obs_buf = torch.zeros((self.num_envs, self.num_obs + self.num_commands), device=self.device, dtype=gs.tc_float)
        self.obs_buf = torch.zeros(self.num_envs, (self.num_obs * self.num_stacks) + self.num_commands, device=self.device, dtype=gs.tc_float)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs, (self.num_privileged_obs * self.num_stacks) + self.num_commands, device=self.device, dtype=gs.tc_float)
            self.extras = {"observations": {"critic" : self.privileged_obs_buf}}
        else:
            self.extras = {"observations": {}}
            self.privileged_obs_buf = None
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_int)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_int)
        
        self.create_sim()

    def get_observations(self):
        return self.obs_buf, self.extras
    
    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset_idx(self, env_ids):
        """Reset selected robots"""
        raise NotImplementedError

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))  
        obs, extras = self.get_observations()
        # obs, _, _, extras = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, extras

    def step(self, actions):
        raise NotImplementedError