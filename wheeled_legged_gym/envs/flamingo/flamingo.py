
import torch
from wheeled_legged_gym.envs.base.wheeled_legged_robot import LeggedRobot
from wheeled_legged_gym.utils.gs_utils import *

class Flamingo(LeggedRobot):
    """ Flamingo class """

    def compute_single_observations(self):
        """ Computes single observations
            Returns a tensor of shape (num_envs, num_obs)
        """
        self.single_obs_buf = torch.cat((
            (self.dof_pos[:, self.p_indices] - self.default_dof_pos[self.p_indices]) * self.obs_scales.dof_pos,   # num_dofs
            self.dof_vel * self.obs_scales.dof_vel,                            # num_dofs
            self.base_ang_vel * self.obs_scales.ang_vel,                       # 3
            self.projected_gravity * self.obs_scales.gravity,                  # 3
            self.actions                                                       # num_actions
        ), dim=-1)

    def compute_single_priv_observations(self):
        """ Computes single privileged observations
            Returns a tensor of shape (num_envs, num_privileged_obs)
        """
        self.single_priv_obs_buf = torch.cat((
            (self.dof_pos[:, self.p_indices] - self.default_dof_pos[self.p_indices]) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.actions,
        ),
        dim=-1
        )

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.ones(self.num_obs + self.num_commands, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:6] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[6:14] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[14:17] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[17:20] = noise_scales.gravity * noise_level * self.obs_scales.gravity
        noise_vec[20:28] = 0. # actions
        noise_vec[28:31] = 0. # commands
        if self.cfg.terrain.measure_heights:
            noise_vec[31:218] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements

        return noise_vec

    #------------ reward functions----------------
    def _reward_shoulder_align(self):
        """ Computes the reward for shoulder alignment
            Returns a tensor of shape (num_envs, 1)
        """
        left_idx = self.get_joint_name_idx("left_shoulder_joint")
        right_idx = self.get_joint_name_idx("right_shoulder_joint")
        penalty = torch.square(self.dof_pos[:, left_idx] - self.dof_pos[:, right_idx])
        return penalty
    
    def _reward_hip_deviations(self):
        """ Computes the reward for hip joint deviations
            Returns a tensor of shape (num_envs, 1)
        """
        indices = self.get_joint_name_idx(".*_hip_joint")
        penalty = torch.sum(torch.square(self.dof_pos[:, indices] - self.default_dof_pos[indices]), dim=1)
        return penalty
    
    def _reward_shoulder_deviations(self):
        """Computes the reward for shoulder joint deviations.
        Returns a tensor of shape (num_envs, 1)
        """
        indices = self.get_joint_name_idx(".*_shoulder_joint")
        penalty = torch.sum(torch.square(self.dof_pos[:, indices] - self.default_dof_pos[indices]), dim=1)
        return penalty