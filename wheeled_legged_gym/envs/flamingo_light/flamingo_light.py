
import torch
from wheeled_legged_gym.envs.base.wheeled_legged_robot import LeggedRobot
from wheeled_legged_gym.utils.gs_utils import *

class FlamingoLight(LeggedRobot):
    def compute_single_observations(self):
        """ Computes single observations
            Returns a tensor of shape (num_envs, num_obs)
        """
        self.single_obs_buf = torch.cat((
            (self.dof_pos[:, :2] - self.default_dof_pos[:2]) * self.obs_scales.dof_pos,   # num_dofs
            self.dof_vel * self.obs_scales.dof_vel,                            # num_dofs
            self.base_ang_vel * self.obs_scales.ang_vel,                       # 3
            self.projected_gravity,                                            # 3
            self.actions                                                       # num_actions
        ), dim=-1)

    def compute_single_priv_observations(self):
        """ Computes single privileged observations
            Returns a tensor of shape (num_envs, num_privileged_obs)
        """
        self.single_priv_obs_buf = torch.cat((
            (self.dof_pos[:, :2] - self.default_dof_pos[:2]) * self.obs_scales.dof_pos,
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
        noise_vec[:2] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[2:6] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[6:9] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[9:12] = noise_scales.gravity * noise_level  * self.obs_scales.gravity
        noise_vec[12:16] = 0. # actions
        noise_vec[16:20] = 0. # commands
        if self.cfg.terrain.measure_heights:
            noise_vec[20:207] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements

        return noise_vec