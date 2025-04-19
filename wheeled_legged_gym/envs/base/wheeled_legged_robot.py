import genesis as gs
from genesis.utils.geom import transform_by_quat, inv_quat, transform_quat_by_quat, quat_to_xyz
from genesis.utils import *
import numpy as np
import os
import textwrap
from prettytable import PrettyTable
import re
import torch

from wheeled_legged_gym import WHEELED_LEGGED_GYM_ROOT_DIR
from wheeled_legged_gym.envs.base.base_task import BaseTask
from wheeled_legged_gym.utils.math import wrap_to_pi, torch_rand_sqrt_float, quat_apply_yaw
from wheeled_legged_gym.utils.terrain import Terrain
from wheeled_legged_gym.utils.helpers import class_to_dict
from wheeled_legged_gym.utils.gs_utils import *
from .wheeled_legged_robot_config import WheeledLeggedRobotCfg

class LeggedRobot(BaseTask):
    def __init__(self, cfg: WheeledLeggedRobotCfg, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.height_samples = None
        self.debug_viz = self.cfg.env.debug_viz
        self.init_done = False
        self._parse_cfg(self.cfg)

        super().__init__(self.cfg, sim_device, headless)

        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions

        if self.cfg.control.use_implicit_controller: # use embedded pd controller
            self.torques = self._compute_torques(exec_actions) # calculate torque with explicit pd controller
            # self.torques = self.robot.get_dofs_force(self.motor_dofs) # It is already clipped
            if self.has_p:
                target_dof_pos = self._compute_target_dof_pos(exec_actions[:, self.p_indices])
                self.robot.control_dofs_position(target_dof_pos, self.p_motor_dofs)
            if self.has_v:
                target_dof_vel = self._compute_target_dof_vel(exec_actions[:, self.v_indices])
                self.robot.control_dofs_velocity(target_dof_vel, self.v_motor_dofs)
            self.scene.step()
        else:
            for _ in range(self.cfg.control.decimation): # use self-implemented pd controller
                self.torques = self._compute_torques(exec_actions)
                torques_clipped = torch.clip(self.torques, -(self.torque_limits.repeat(self.num_envs, 1)), self.torque_limits.repeat(self.num_envs, 1))
                if self.num_build_envs == 0:
                    torques = torques_clipped.squeeze()
                    self.robot.control_dofs_force(torques, self.motor_dofs)
                else:
                    self.robot.control_dofs_force(torques_clipped, self.motor_dofs)
                self.scene.step()
                self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
                self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)

        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
            
            # Todo: Handle privileged observations when num_privileged_obs is None / While configuring priv obs using none noisy single obs buf 
            if "critic" in self.extras["observations"]:
                self.extras["observations"]["critic"] = self.privileged_obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # compute observations, rewards, resets, ...
        self._get_observations()
        self._post_physics_step_callback()
        self.compute_observations()

        # compute rew_buf
        self.compute_reward()
        
        # compute reset_buf
        self.check_termination()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        # if self.num_build_envs > 0:
        self.reset_idx(env_ids)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        if self.debug_viz:
            self._draw_debug_vis()

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.link_contact_forces[:, self.termination_indices, :], dim=-1)> 1.0, dim=1)
        self.time_out_buf = self.episode_length_buf >= self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        
    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)
        
        # domain randomization
        if self.cfg.domain_rand.randomize_friction:
            self._randomize_friction(env_ids)
        if self.cfg.domain_rand.randomize_base_mass:
            self._randomize_base_mass(env_ids)
        if self.cfg.domain_rand.randomize_com_displacement:
            self._randomize_com_displacement(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
            # NaN 또는 Inf 검출 및 0 강제 적용
            if self.debug:
                if torch.isnan(rew).any():
                    nan_idx = torch.nonzero(torch.isnan(rew), as_tuple=False)
                    print(f"[Error] Reward '{name}' has NaN at indices: {nan_idx}")
                    rew[torch.isnan(rew)] = 0.0
                if torch.isinf(rew).any():
                    inf_idx = torch.nonzero(torch.isinf(rew), as_tuple=False)
                    print(f"[Error] Reward '{name}' has Inf at indices: {inf_idx}")
                    rew[torch.isinf(rew)] = 0.0
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def _get_observations(self):
        # prepare quantities
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler[:] = quat_to_xyz(transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat), degrees=False)
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = gs_transform_by_quat(self.robot.get_vel(), inv_base_quat) # trasform to base frame
        self.base_ang_vel[:] = gs_transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity[:] = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)
        self.link_contact_forces[:] = self.robot.get_links_net_contact_force().clone().detach().to(self.device, dtype=self.precision)
    
    def compute_single_observations(self):
        """ Computes single observations
            Returns a tensor of shape (num_envs, num_obs)
        """
        self.single_obs_buf = torch.cat((
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,   # num_dofs
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
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity * self.obs_scales.gravity,
            self.actions,
        ),
        dim=-1
        )

    def compute_observations(self):
        """ Computes observations and stacks them according to num_stack.
        """
        # 1. Compute single observations
        self.compute_single_observations()
        if self.privileged_obs_buf is not None:
            self.compute_single_priv_observations()

        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.base_pos[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) \
                    * self.obs_scales.height_measurements
            self.single_obs_buf = torch.cat((self.single_obs_buf, heights), dim=-1)

        if self.add_noise:
            assert self.noise_scale_vec.shape[0] >= self.num_obs + self.num_commands, "noise_scale_vec shape mismatch"

            # Todo: Generalize noise vec with commands
            self.single_obs_buf[:] += (2 * torch.rand_like(self.single_obs_buf) - 1) * self.noise_scale_vec[:self.num_obs]
            self.commands[:] += (2 * torch.rand_like(self.commands) - 1) * self.noise_scale_vec[self.num_obs:self.num_obs + self.num_commands]

        # 2. Init stacked_obs_buf (only once)
        if not hasattr(self, 'stacked_obs_buf'):
            assert self.single_obs_buf.shape[1] == self.num_obs, "single_obs shape mismatch"
            # stacked_obs_buf shape: (num_envs, num_stack * num_obs + num_commands)
            self.stacked_obs_buf = self.single_obs_buf.new_zeros(self.num_envs, self.num_stacks *  self.num_obs)

        if self.privileged_obs_buf is not None:
            if not hasattr(self, 'stacked_priv_obs_buf'):
                assert self.single_priv_obs_buf.shape[1] == self.num_privileged_obs, "single_priv_obs shape mismatch"
                # stacked_obs_buf shape: (num_envs, num_stack * num_obs + num_commands)
                self.stacked_priv_obs_buf = self.single_obs_buf.new_zeros(self.num_envs, self.num_stacks *  self.num_privileged_obs)
            
        # 3. Update stacked_obs_buf
        reset_mask = self.episode_length_buf.eq(0)
        if reset_mask.any():
            self.stacked_obs_buf[reset_mask] = self.single_obs_buf[reset_mask].repeat(1, self.num_stacks)
            if self.privileged_obs_buf is not None:
                self.stacked_priv_obs_buf[reset_mask] = self.single_priv_obs_buf[reset_mask].repeat(1, self.num_stacks)
        non_reset_mask = ~reset_mask
        if non_reset_mask.any():
            self.stacked_obs_buf[non_reset_mask] = torch.roll(self.stacked_obs_buf[non_reset_mask],
                                                            shifts=self.num_obs,
                                                            dims=1)
            self.stacked_obs_buf[non_reset_mask, :self.num_obs] = self.single_obs_buf[non_reset_mask]
            if self.privileged_obs_buf is not None:
                self.stacked_priv_obs_buf[non_reset_mask] = torch.roll(self.stacked_priv_obs_buf[non_reset_mask],
                                                                    shifts=self.num_privileged_obs,
                                                                    dims=1)
                self.stacked_priv_obs_buf[non_reset_mask, :self.num_privileged_obs] = self.single_priv_obs_buf[non_reset_mask]
        
        # 4. Cat stacked_obs with commands
        self.obs_buf = torch.cat([self.stacked_obs_buf, self.commands * self.commands_scale], dim=-1)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.cat([self.stacked_priv_obs_buf, self.commands * self.commands_scale], dim=-1)
        else:
            self.privileged_obs_buf = self.obs_buf

        
    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.sim_dt, 
                substeps=self.sim_substeps
            ),
            # vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(self.cfg.viewer.num_rendered_envs))),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(1 / self.dt * self.cfg.control.decimation),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.sim_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                enable_self_collision=self.cfg.asset.self_collisions,
            ),
            show_viewer= not self.headless,
            show_FPS=False,
        )
            
        # add camera if needed
        if self.cfg.viewer.add_camera:
            self._setup_camera()
        
        # add terrain
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type=='plane':
            self.terrain = self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
            self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        elif mesh_type=='heightfield':
            self.utils_terrain = Terrain(self.cfg.terrain)
            self._create_heightfield()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self.terrain.set_friction(self.cfg.terrain.friction)
        # specify the boundary of the heightfield
        self.terrain_x_range = torch.zeros(2, device=self.device)
        self.terrain_y_range = torch.zeros(2, device=self.device)
        if self.cfg.terrain.mesh_type=='heightfield':
            self.terrain_x_range[0] = -self.cfg.terrain.border_size + 1.0 # give a small margin(1.0m)
            self.terrain_x_range[1] = self.cfg.terrain.border_size + self.cfg.terrain.num_rows * self.cfg.terrain.terrain_length - 1.0
            self.terrain_y_range[0] = -self.cfg.terrain.border_size + 1.0
            self.terrain_y_range[1] = self.cfg.terrain.border_size + self.cfg.terrain.num_cols * self.cfg.terrain.terrain_width - 1.0
        # elif self.cfg.terrain.mesh_type=='plane': # the plane used has limited size, 
        #                                           # and the origin of the world is at the center of the plane
        #     self.terrain_x_range[0] = -self.cfg.terrain.plane_length/2+1
        #     self.terrain_x_range[1] = self.cfg.terrain.plane_length/2-1
        #     self.terrain_y_range[0] = -self.cfg.terrain.plane_length/2+1 # the plane is a square
        #     self.terrain_y_range[1] = self.cfg.terrain.plane_length/2-1
        self._create_envs()
    
    def set_camera(self, pos, lookat):
        """ Set camera position and direction
        """
        self.floating_camera.set_pose(
            pos=pos,
            lookat=lookat
        )
    
    #------------- Callbacks --------------
    def _setup_camera(self):
        ''' Set camera position and direction
        '''
        self.floating_camera = self.scene.add_camera(
            res = (1280, 960),
            pos=np.array(self.cfg.viewer.pos),
            lookat=np.array(self.cfg.viewer.lookat),
            fov=40,
            GUI=True,
        )

        self._recording = False
        self._recorded_frames = []
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = gs_transform_by_quat(self.forward_vec, self.base_quat)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots:
            self._push_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = gs_rand_float(*self.cfg.commands.ranges.lin_vel_x, (len(env_ids),), self.device, dtype=self.precision)
        self.commands[env_ids, 1] = gs_rand_float(*self.cfg.commands.ranges.lin_vel_y, (len(env_ids),), self.device, dtype=self.precision)
        self.commands[env_ids, 2] = gs_rand_float(*self.cfg.commands.ranges.ang_vel_yaw, (len(env_ids),), self.device, dtype=self.precision)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        # Todo : handle pos/vel control in case of no p or v indices
        target_pos = self._compute_target_dof_pos(actions[:, self.p_indices])  # shape: (batch, len(p_indices))
        error_p = target_pos - self.dof_pos[:, self.p_indices]
        torque_p = self.batched_p_gains[:, self.p_indices] * error_p - self.batched_d_gains[:, self.p_indices] * self.dof_vel[:, self.p_indices]
        
        target_vel = self._compute_target_dof_vel(actions[:, self.v_indices])  # shape: (batch, len(v_indices))
        error_v = target_vel - self.dof_vel[:, self.v_indices]
        torque_v = self.batched_d_gains[:, self.v_indices] * error_v

        torque = torch.zeros((self.num_envs, len(self.motor_dofs)), device=self.device, dtype=self.dof_pos.dtype)
        torque[:, self.p_indices] = torque_p
        torque[:, self.v_indices] = torque_v

        return torque

    def _compute_target_dof_pos(self, actions):
        actions_scaled_p = actions * self.cfg.control.control_p.action_scale
        target_dof_pos = actions_scaled_p + self.default_dof_pos[self.p_indices]
        return target_dof_pos

    def _compute_target_dof_vel(self, actions):
        action_scaled_v = actions * self.cfg.control.control_v.action_scale
        return action_scaled_v

    def _reset_dofs(self, envs_idx):
        """ Resets DOF position and velocities of selected environments. """
        all_dof_pos = torch.zeros((self.num_envs, len(self.all_dofs)), device=self.device, dtype=self.precision)

        if self.cfg.init_state.randomize_joint:
            rmin, rmax = self.cfg.init_state.randomize_joint.get("range")

            for i, name in enumerate(self.dof_names):  # i는 motor_dofs 기준 인덱스
                local_idx = self.motor_dofs_local[i]   # all_dof_pos의 column 위치

                if self.cfg.init_state.randomize_joint.get(name, False):
                    noise = gs_rand_float(rmin, rmax, (len(envs_idx),), device=self.device, dtype=self.precision)
                else:
                    noise = torch.zeros((len(envs_idx),), device=self.device, dtype=all_dof_pos.dtype)

                default = self.cfg.init_state.default_joint_angles[name]
                all_dof_pos[envs_idx, local_idx] = default + noise
        else:
            for i, name in enumerate(self.dof_names):
                local_idx = self.motor_dofs_local[i]
                default = self.cfg.init_state.default_joint_angles[name]
                all_dof_pos[envs_idx, local_idx] = default

        self.robot.set_dofs_position(
            position=all_dof_pos[envs_idx],
            dofs_idx_local=self.all_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        self.dof_vel[envs_idx] = 0.0

        # reset dof velocity to zero
        self.robot.zero_all_dofs_velocity(envs_idx)
        
    def _reset_root_states(self, envs_idx):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base pos: xy [-1, 1]
        if self.custom_origins:   
            self.base_pos[envs_idx] = self.base_init_pos
            self.base_pos[envs_idx] += self.env_origins[envs_idx]
            self.base_pos[envs_idx, :2] += gs_rand_float(-1.0, 1.0, (len(envs_idx), 2), self.device, dtype=self.precision)
        else:
            self.base_pos[envs_idx] = self.base_init_pos
            self.base_pos[envs_idx] += self.env_origins[envs_idx]

        # Reset base position and orientation.
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)

        roll  = gs_rand_uniform_float(*self.cfg.init_state.ranges.roll, len(envs_idx), self.device, dtype=self.precision)
        pitch = gs_rand_uniform_float(*self.cfg.init_state.ranges.pitch, len(envs_idx), self.device, dtype=self.precision)
        yaw   = gs_rand_uniform_float(*self.cfg.init_state.ranges.yaw, len(envs_idx), self.device, dtype=self.precision)
        base_euler = torch.stack([roll, pitch, yaw], dim=-1)  # shape: (N, 3)

        self.base_quat[envs_idx] = transform_quat_by_quat(gs.euler_to_quat(base_euler), self.base_quat[envs_idx])  

        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)

        transformed_quat = transform_quat_by_quat(torch.ones_like(self.base_quat[envs_idx]) * self.inv_base_init_quat,self.base_quat[envs_idx])
        self.base_euler[envs_idx] = quat_to_xyz(transformed_quat, degrees=False)
        self.projected_gravity[envs_idx] = transform_by_quat(self.global_gravity[envs_idx], inv_quat(self.base_quat[envs_idx]))
        
        # reset root states - linear velocity
        lin_x = gs_rand_uniform_float(*self.cfg.init_state.ranges.lin_vel_x, len(envs_idx), self.device, dtype=self.precision)
        lin_y = gs_rand_uniform_float(*self.cfg.init_state.ranges.lin_vel_y, len(envs_idx), self.device, dtype=self.precision)
        lin_z = gs_rand_uniform_float(*self.cfg.init_state.ranges.lin_vel_z, len(envs_idx), self.device, dtype=self.precision)
        self.base_lin_vel[envs_idx] = torch.stack([lin_x, lin_y, lin_z], dim=-1).to(device=self.device, dtype=self.precision)
        # reset root states - angular velocity
        ang_x = gs_rand_uniform_float(*self.cfg.init_state.ranges.ang_vel_x, len(envs_idx), self.device, dtype=self.precision)
        ang_y = gs_rand_uniform_float(*self.cfg.init_state.ranges.ang_vel_y, len(envs_idx), self.device, dtype=self.precision)
        ang_z = gs_rand_uniform_float(*self.cfg.init_state.ranges.ang_vel_z, len(envs_idx), self.device, dtype=self.precision)
        self.base_ang_vel[envs_idx] = torch.stack([ang_x, ang_y, ang_z], dim=-1).to(device=self.device, dtype=self.precision)
        
        base_vel = torch.concat([self.base_lin_vel[envs_idx], self.base_ang_vel[envs_idx]], dim=1)
        self.robot.set_dofs_velocity(velocity=base_vel, dofs_idx_local=self.base_dofs, envs_idx=envs_idx)


    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        if self.push_interval_s > 0 and not self.debug:
            max_push_vel_xy = self.cfg.domain_rand.max_push_vel_xy
            # in Genesis, base link also has DOF, it's 6DOF if not fixed.
            dofs_vel = self.robot.get_dofs_velocity() # (num_envs, num_dof) [0:3] ~ base_link_vel
            push_vel = gs_rand_float(-max_push_vel_xy, max_push_vel_xy, (self.num_envs, 2), self.device, dtype=self.precision)
            push_vel[((self.common_step_counter + self.env_identities) % int(self.push_interval_s / self.dt) != 0)] = 0
            dofs_vel[:, :2] += push_vel
            self.robot.set_dofs_velocity(dofs_vel)

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.base_pos[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.utils_terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)


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
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[36:48] = 0. # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[48:235] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec

    #----------------------------------------

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        self.common_step_counter = 0
        self.base_init_pos = torch.tensor(
            self.cfg.init_state.pos, device=self.device, dtype=self.precision
        )
        self.base_init_quat = torch.tensor(
            self.cfg.init_state.rot, device=self.device, dtype=self.precision
        )
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=self.precision)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=self.precision)
        self.base_euler = torch.zeros((self.num_envs, 3), device=self.device, dtype=self.precision)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=self.precision)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=self.precision).repeat(
            self.num_envs, 1
        )
        self.single_obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=self.precision)
        if self.num_privileged_obs is not None:
            self.single_priv_obs_buf = torch.zeros((self.num_envs, self.num_privileged_obs), device=self.device, dtype=self.precision)
        else:
            self.single_priv_obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=self.precision)
        self.obs_buf = torch.zeros((self.num_envs, (self.num_obs * self.num_stacks) + self.num_commands), device=self.device, dtype=self.precision)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=self.precision)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=self.precision)
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], 
            device=self.device,
            dtype=self.precision, 
            requires_grad=False,) # TODO change this
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=self.precision)
        self.last_actions = torch.zeros_like(self.actions)

        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=self.precision)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=self.precision)
        self.feet_air_time = torch.zeros((self.num_envs, len(self.feet_indices)), device=self.device, dtype=self.precision)
        self.last_feet_contacts = torch.zeros((self.num_envs, len(self.feet_indices)), device=self.device,dtype=gs.tc_int)
        self.link_contact_forces = torch.zeros(
            (self.num_envs, self.robot.n_links, 3), device=self.device, dtype=self.precision
        )
        self.continuous_push = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=self.precision
        )
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.forward_vec = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=self.precision
        )
        self.forward_vec[:, 0] = 1.0
        self.env_identities = torch.arange(
            self.num_envs,
            device=self.device,
            dtype=gs.tc_int, 
        )
        self.terrain_heights = torch.zeros(
            (self.num_envs,),
            device=self.device,
            dtype=self.precision,
        )
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        else:
            self.measured_heights = 0

        self.default_dof_pos = torch.tensor(
            [self.cfg.init_state.default_joint_angles[name] for name in self.cfg.asset.dof_names],
            device=self.device,
            dtype=self.precision,
        )

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=self.precision, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        self.terrain = self.scene.add_entity(
            gs.morphs.Terrain(
                pos=(-self.cfg.terrain.border_size, -self.cfg.terrain.border_size, 0.0),
                horizontal_scale=self.cfg.terrain.horizontal_scale,
                vertical_scale=self.cfg.terrain.vertical_scale,
                height_field=self.utils_terrain.height_field_raw,
            )
        )
        self.height_samples = torch.tensor(self.utils_terrain.heightsamples).view(
            self.utils_terrain.tot_rows, self.utils_terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """Creates environments:
        1. Loads the robot URDF/MJCF asset and creates the entity.
        2. Stores indices of different bodies of the robot.
        """
        asset_path = self.cfg.asset.file.format(WHEELED_LEGGED_GYM_ROOT_DIR=WHEELED_LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)
        asset_type = self.cfg.asset.asset_type
        assert asset_type in ["urdf", "mjcf"], f"Unknown asset type: {asset_type}. Please check the config file."
        if asset_type == "urdf":
            self.robot = self.scene.add_entity(
                gs.morphs.URDF(
                    file=os.path.join(asset_root, asset_file),
                    merge_fixed_links=True,
                    links_to_keep=self.cfg.asset.links_to_keep,
                    pos=np.array(self.cfg.init_state.pos),
                    quat=np.array(self.cfg.init_state.rot),
                    fixed=self.cfg.asset.fix_base_link,
                ),
                visualize_contact=self.debug,
                vis_mode="visual",
            )
        elif asset_type == "mjcf":
            self.robot = self.scene.add_entity(
                gs.morphs.MJCF(
                    file=os.path.join(asset_root, asset_file),
                    convexify=self.cfg.asset.convexify,
                    pos=np.array(self.cfg.init_state.pos),
                    quat=np.array(self.cfg.init_state.rot),
                ),
                visualize_contact=self.debug,
                vis_mode="visual",
            )
        
        # Build the scene for the given number of environments.
        self.scene.build(n_envs=self.num_envs)
        
        self._get_env_origins()

        # Get All motor dof indices
        self.all_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.all_dof_names]
        # Get Active motor dof indices.
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.dof_names]
        # Get Active motor dof indices in all_dofs.
        self.motor_dofs_local = [self.all_dofs.index(i) for i in self.motor_dofs]
        # Get Base free joint
        assert len(self.base_dof_names) == 1, f"Base joint name should be unique. Found {len(self.base_dof_names)} base joints."
        if self.cfg.asset.asset_type == "urdf":
            self.base_dofs = [0, 1, 2, 3, 4, 5]
        elif self.cfg.asset.asset_type == "mjcf":
            self.base_dofs = self.robot.get_joint(self.base_dof_names[0]).dof_idx_local
        # Get active joint idx from name 
        self.joint_name_to_idx = {name: idx for idx, name in enumerate(self.dof_names)}

        # Randomize friction, base mass, and COM displacement if configured.
        self._randomize_friction()
        self._randomize_base_mass()
        self._randomize_com_displacement()

        # Set up control indices.
        self.p_indices = []      # Relative indices for dof_pos.
        self.v_indices = []      # Relative indices for dof_vel.
        self.p_motor_dofs = []   # Absolute indices for motor position control.
        self.v_motor_dofs = []   # Absolute indices for motor velocity control.
        
        for idx, name in enumerate(self.cfg.asset.dof_names):
            if any(key in name for key in self.cfg.control.control_p.stiffness.keys()):
                self.p_indices.append(idx)
                self.p_motor_dofs.append(self.motor_dofs[idx])
            elif any(key in name for key in self.cfg.control.control_v.stiffness.keys()):
                self.v_indices.append(idx)
                self.v_motor_dofs.append(self.motor_dofs[idx])
            else:
                raise ValueError(f"Unknown dof name: {name}. Please check the config file.")

        # Flags for control modes.
        self.has_p = len(self.p_motor_dofs) > 0
        self.has_v = len(self.v_motor_dofs) > 0
            
        # Helper: get dof attributes.
        def get_dof_info(dof_name):
            # 먼저 control_p, 다음 control_v 순으로 확인.
            for mode in ['control_p', 'control_v']:
                mode_cfg = getattr(self.cfg.control, mode)
                for key in mode_cfg.stiffness.keys():
                    if key in dof_name:
                        return mode_cfg.stiffness[key], mode_cfg.damping[key], mode_cfg.torque_limit[key], mode_cfg.vel_limit[key]
            return 0.0, 0.0  # torque_limit이 없는 경우
        
        info = [get_dof_info(name) for name in self.cfg.asset.dof_names]
        p_gains, d_gains, torque_limits, vel_limits = zip(*info)        

        # Convert attributes to Tensor and batch.
        self.p_gains = torch.tensor(p_gains, device=self.device, dtype=self.precision)
        self.d_gains = torch.tensor(d_gains, device=self.device, dtype=self.precision)
        self.batched_p_gains = self.p_gains.unsqueeze(0).repeat(self.num_envs, 1)
        self.batched_d_gains = self.d_gains.unsqueeze(0).repeat(self.num_envs, 1)
        
        # Set PD gains on the robot.
        self.robot.set_dofs_kp(self.p_gains, self.motor_dofs)
        self.robot.set_dofs_kv(self.d_gains, self.motor_dofs)

        self.dof_vel_limits = torch.tensor(vel_limits, device=self.device, dtype=self.precision) 

        self.robot.set_dofs_force_range(
            lower=-np.array(torque_limits),
            upper=np.array(torque_limits),
            dofs_idx_local=self.motor_dofs
        )

        # Get link indices.
        def find_link_indices(names):
            link_indices = []
            for link in self.robot.links:
                if any(name in link.name for name in names):
                    link_indices.append(link.idx - self.robot.link_start)
            return link_indices

        self.termination_indices = find_link_indices(self.cfg.asset.terminate_after_contacts_on)
        all_link_names = [link.name for link in self.robot.links]
        self.penalized_indices = find_link_indices(self.cfg.asset.penalize_contacts_on)
        self.feet_indices = find_link_indices(self.cfg.asset.foot_name)        
        self.feet_link_indices_world_frame = [i+1 for i in self.feet_indices]
        
        # DOF position limits.
        self.dof_pos_limits = torch.stack(self.robot.get_dofs_limit(self.motor_dofs), dim=1)
        # Todo: handle torque limits above
        self.torque_limits = self.robot.get_dofs_force_range(self.motor_dofs)[1]

        for i in range(self.dof_pos_limits.shape[0]):
            m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
            r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
            self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
            self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit


        # --- Joint Index Mapping Table ---
        joint_index_table = PrettyTable()
        joint_index_table.field_names = ["Joint Name", "Idx (local)", "Idx (all_dof)"]

        for name in self.dof_names:
            local_idx = self.robot.get_joint(name).dof_idx_local
            all_dof_idx = self.all_dofs.index(local_idx)
            joint_index_table.add_row([name, local_idx, all_dof_idx])

        print(joint_index_table)

        # --- Joint Attributes ---
        joint_table = PrettyTable()
        joint_table.field_names = ["Joint Name", "p_gain", "d_gain", "Tau Lim", "Range", "Vel Lim", "Mode"]
        for idx, name in enumerate(self.dof_names):
            if idx in self.p_indices:
                mode = "position"
            elif idx in self.v_indices:
                mode = "velocity"
            else:
                mode = "None"
            dof_min, dof_max = self.dof_pos_limits[idx].cpu().tolist()
            dof_range = f"[{dof_min:.4f}, {dof_max:.4f}]"
            joint_table.add_row([name, p_gains[idx], d_gains[idx], torque_limits[idx], dof_range, vel_limits[idx], mode])
        print(joint_table)


        # --- Joint Names ---
        all_joint_table = PrettyTable()
        all_joint_table.field_names = ["Joint Names"]
        wrapped_all_joint_names = "\n".join(textwrap.wrap(", ".join(self.all_dof_names), width=40))
        all_joint_table.add_row([wrapped_all_joint_names])
        print(all_joint_table)

        # --- Link Names ---
        all_link_table = PrettyTable()
        all_link_table.field_names = ["All Link Names"]
        wrapped_all_link_names = "\n".join(textwrap.wrap(", ".join(all_link_names), width=40))
        all_link_table.add_row([wrapped_all_link_names])
        print(all_link_table)

        # --- Link Information ---
        link_table = PrettyTable()
        link_table.field_names = ["Link Type", "Indices"]
        link_table.add_row(["Termination Link Indices", str(self.termination_indices)])
        link_table.add_row(["Penalized Link Indices", str(self.penalized_indices)])
        link_table.add_row(["Feet Link Indices", str(self.feet_indices)])
        print(link_table)

    def _randomize_friction(self, env_ids=None):
            ''' Randomize friction of all links for the specified environments. '''
            if env_ids is None:
                env_ids = torch.arange(self.num_envs, device=self.device)
            # 생성할 텐서의 첫 번째 차원은 env_ids의 길이로 처리합니다.
            friction_ratio = gs_rand_uniform_float(*self.cfg.domain_rand.friction_range,
                                                    shape=(len(env_ids), self.robot.n_links),
                                                    device=self.device, dtype=self.precision)
            # env_ids에 해당하는 환경들에 적용하도록 envs_idx 인자를 추가합니다.
            self.robot.set_friction_ratio(
                friction_ratio=friction_ratio,
                ls_idx_local=np.arange(0, self.robot.n_links),
                envs_idx=env_ids
            )

    def _randomize_base_mass(self, env_ids=None):
        ''' Randomize base mass for the specified environments. '''
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        found = False
        for link in self.robot.links:
            if 'base' in link.name:
                found = True
                base_link_idx = link.idx - self.robot.link_start
                break  # base_link을 찾으면 바로 종료
        if not found:
            raise KeyError("No link with the name 'base_link' exists.")

        mass_shift = gs_rand_uniform_float(*self.cfg.domain_rand.added_mass_range,
                                            shape=(len(env_ids), 1),
                                            device=self.device, dtype=self.precision)
        self.robot.set_mass_shift(
            mass_shift=mass_shift,
            ls_idx_local=np.array([base_link_idx]),
            envs_idx=env_ids
        )

    def _randomize_com_displacement(self, env_ids=None):
        ''' Randomize center-of-mass (COM) displacement for the specified environments. '''
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        com_shift = gs_rand_uniform_float(*self.cfg.domain_rand.com_displacement_range,
                                            shape=(len(env_ids), self.robot.n_links, 3),
                                            device=self.device, dtype=self.precision)
        self.robot.set_COM_shift(
            com_shift=com_shift,
            ls_idx_local=np.arange(0, self.robot.n_links),
            envs_idx=env_ids
        )

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.dt
        if self.cfg.control.use_implicit_controller: # use embedded PD controller
            self.sim_dt = self.dt
            self.sim_substeps = self.cfg.control.decimation
        else: # use explicit PD controller
            self.sim_dt = self.dt / self.cfg.control.decimation
            self.sim_substeps = 1
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.push_interval_s = self.cfg.domain_rand.push_interval_s

        self.all_dof_names = self.cfg.asset.all_dof_names if self.cfg.asset.all_dof_names is not None else self.cfg.asset.dof_names
        self.dof_names = self.cfg.asset.dof_names
        self.base_dof_names = self.cfg.asset.base_dof_names
        self.simulate_action_latency = self.cfg.domain_rand.simulate_action_latency
        self.debug = self.cfg.env.debug

        """ Sets the precision of the simulation. """
        self.precision = self.cfg.sim.precision
        if self.precision == '32':
            self.precision = gs.tc_float
        elif self.precision == '64':
            self.precision = torch.double
        else:
            raise ValueError(f"Unknown precision: {self.cfg.precision}")
        
    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height points
        if not self.cfg.terrain.measure_heights:
            return
        self.scene.clear_debug_objects()
        height_points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points)
        height_points[0, :, 0] += self.base_pos[0, 0]
        height_points[0, :, 1] += self.base_pos[0, 1]
        height_points[0, :, 2] = self.measured_heights[0, :]
        # print(f"shape of height_points: ", height_points.shape) # (num_envs, num_points, 3)
        self.scene.draw_debug_spheres(height_points[0, :], radius=0.03, color=(0, 0, 1, 0.7)) # only draw for the first env

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.utils_terrain.env_origins).to(self.device).to(torch.double)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            # num_cols = np.floor(np.sqrt(self.num_envs))
            # num_rows = np.ceil(self.num_envs / num_cols)
            # xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            # # plane has limited size, we need to specify spacing base on num_envs, to make sure all robots are within the plane
            # # restrict envs to a square of size [plane_length/2, plane_length/2]
            # spacing = min((self.cfg.terrain.plane_length / 2) / (num_rows-1), (self.cfg.terrain.plane_length / 2) / (num_cols-1))
            # self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            # self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            # self.env_origins[:, 2] = 0.
            # self.env_origins[:, 0] -= self.cfg.terrain.plane_length / 4
            # self.env_origins[:, 1] -= self.cfg.terrain.plane_length / 4
    
    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.base_pos[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.base_pos[:, :3]).unsqueeze(1)

        points += self.cfg.terrain.border_size
        points = (points/self.cfg.terrain.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.cfg.terrain.vertical_scale

    def get_joint_name_idx(self, joint_names):
        """
        Returns either a single int or a list of ints matching the given regex pattern.

        Args:
            joint_names (str): Regex pattern for joint names (e.g., '.*_shoulder_joint')

        Returns:
            int or List[int]:  
                - If exactly one joint matches, returns its index (int).  
                - If multiple match, returns a list of indices.  
                - If none match, raises a ValueError.
        """
        matches = [
            idx for name, idx in self.joint_name_to_idx.items()
            if re.fullmatch(joint_names, name)
        ]
        if not matches:
            raise ValueError(f"No joints match pattern '{joint_names}'")
        if len(matches) == 1:
            return matches[0]
        return matches

    #------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.base_pos[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        rew = torch.square(base_height - self.cfg.rewards.base_height_target)
        return rew
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.link_contact_forces[:, self.penalized_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos[:, self.p_indices] - self.dof_pos_limits[self.p_indices, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos[:, self.p_indices] - self.dof_pos_limits[self.p_indices, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
         # Penalize dof velocities too close to the limit
         # clip to max error = 1 rad/s per joint to avoid huge penalties
         return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_feet_air_time(self):
        # Reward long steps
        contact = self.link_contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_dof_close_to_default(self):
        # Penalize dof position deviation from default
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)