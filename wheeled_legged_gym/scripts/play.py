import genesis as gs
import argparse
import yaml
from wheeled_legged_gym import WHEELED_LEGGED_GYM_ROOT_DIR
import os

from wheeled_legged_gym.envs import *
from wheeled_legged_gym.utils import  export_policy_as_jit, export_policy_as_onnx, task_registry, Logger

import numpy as np
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',           type=str, default='flamingo_light')
    parser.add_argument('--headless',       action='store_true', default=False)  # disable visualization by default
    parser.add_argument('--device',         type=str , default="cuda")
    parser.add_argument("--precision",      type=str, default="32")
    parser.add_argument('--seed',           type=int, default=42)
    parser.add_argument('--num_envs',       type=int, default=4096)
    parser.add_argument('--check_point',    type=int, default=-1)
    parser.add_argument('--max_iterations', type=int, default=3000)
    parser.add_argument('--resume',         type=str, default=False)
    parser.add_argument('--load_run',       type=str, default=-1)
    parser.add_argument('-o', '--offline',  action='store_true', default=False)
    parser.add_argument('--debug',          action='store_true', default=False)

    return parser.parse_args()

def play(args):
    if args.device == "cpu":
        device = gs.cpu
    elif args.device == "cuda":
        device = gs.gpu
    else:
        raise ValueError(f"Unknown device: {args.device}")

    gs.init(
        precision=args.precision, # default: 32
        backend=device,
        logging_level='info',
        debug=args.debug,
        theme='dark', # default: light
        seed=args.seed,
    )
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # env
    env_cfg.env.episode_length_s = 5
    env_cfg.env.num_envs = min(args.num_envs, 1)
    env_cfg.env.debug_viz = False
    env_cfg.viewer.add_camera = False  # use a extra camera for moving

    # disable domain randomization
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.asset.fix_base_link = False
    env_cfg.init_state.yaw_angle_range = [0., 0.]

    # velocity range
    env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    env_cfg.commands.ranges.lin_vel_y = [0., 0.]
    env_cfg.commands.ranges.ang_vel_yaw = [0., 0.]
    env_cfg.commands.ranges.heading = [0, 0]

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # load policy
    train_cfg.resume = True
    runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        export_model_path = os.path.join(task_registry.resume_dir, 'exported')
        export_policy_as_jit(
            runner.alg.policy, runner.obs_normalizer, path=export_model_path, filename="policy.pt"
        )
        export_policy_as_onnx(
            runner.alg.policy, normalizer=runner.obs_normalizer, path=export_model_path, filename="policy.onnx"
        )

    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, rews, dones, infos = env.step(actions)

    # logger = Logger(env.dt)
    # robot_index = 0 # which robot is used for logging
    # joint_index = 1 # which joint is used for logging
    # stop_state_log = 500 # number of steps before plotting states
    # stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    
    # # for MOVE_CAMERA
    # camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    # camera_vel = np.array([1., 1., 0.])
    # camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    # # for FOLLOW_ROBOT
    # camera_lookat_follow = np.array(env_cfg.viewer.lookat)
    # camera_deviation_follow = np.array([0., 3., -1.])
    # camera_position_follow = camera_lookat_follow - camera_deviation_follow
    # # for RECORD_FRAMES
    # stop_record = 400
    # if RECORD_FRAMES:
    #     env.floating_camera.start_recording()

    # for i in range(10*int(env.max_episode_length)):
    #     actions = policy(obs.detach())
    #     obs, rews, dones, infos = env.step(actions.detach())
    #     if MOVE_CAMERA:
    #         camera_position += camera_vel * env.dt
    #         env.set_camera(camera_position, camera_position + camera_direction)
    #         env.floating_camera.render()
    #     if FOLLOW_ROBOT:
    #         # refresh where camera looks at(robot 0 base)
    #         camera_lookat_follow = env.base_pos[robot_index, :].cpu().numpy()
    #         # refresh camera's position
    #         camera_position_follow = camera_lookat_follow - camera_deviation_follow
    #         env.set_camera(camera_position_follow, camera_lookat_follow)
    #         env.floating_camera.render()
    #     if RECORD_FRAMES and i == stop_record:
    #         env.floating_camera.stop_recording(save_to_filename="go2_rough_demo.mp4", fps=30)
    #         print("Saved recording to " + "go2_rough_demo.mp4")
        
    #     # print debug info
    #     # print("base lin vel: ", env.base_lin_vel[robot_index, :].cpu().numpy())
    #     # print("base yaw angle: ", env.base_euler[robot_index, 2].item())
        
    #     if i < stop_state_log:
    #         logger.log_states(
    #             {
    #                 'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
    #                 'dof_pos': env.dof_pos[robot_index, joint_index].item(),
    #                 'dof_vel': env.dof_vel[robot_index, joint_index].item(),
    #                 'dof_torque': env.torques[robot_index, joint_index].item(),
    #                 'command_x': env.commands[robot_index, 0].item(),
    #                 'command_y': env.commands[robot_index, 1].item(),
    #                 'command_yaw': env.commands[robot_index, 2].item(),
    #                 'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
    #                 'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
    #                 'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
    #                 'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
    #                 'contact_forces_z': env.link_contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
    #             }
    #         )
    #     elif i==stop_state_log:
    #         logger.plot_states()
        # if  0 < i < stop_rew_log:
        #     if infos["episode"]:
        #         num_episodes = torch.sum(env.reset_buf).item()
        #         if num_episodes>0:
        #             logger.log_rewards(infos["episode"], num_episodes)
        # elif i==stop_rew_log:
        #     logger.print_rewards()

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False  # only record frames in extra camera view
    MOVE_CAMERA   = True
    FOLLOW_ROBOT  = False
    assert not (MOVE_CAMERA and FOLLOW_ROBOT), "Cannot move camera and follow robot at the same time"
    args = get_args()
    play(args)
