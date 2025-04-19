import os
import argparse
import genesis as gs
from wheeled_legged_gym.envs import *
from wheeled_legged_gym.utils import task_registry, class_to_dict
from wheeled_legged_gym.utils.io import dump_yaml


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
    parser.add_argument('--logger',         type=str, default=None)
    parser.add_argument('-o', '--offline',  action='store_true', default=False)
    parser.add_argument('--debug',          action='store_true', default=False)

    return parser.parse_args()

def train(args):
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

    # print info
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    print(f"Start training for task: {args.task}")
    
    runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)

    dump_yaml(os.path.join(runner.log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(runner.log_dir, "params", "agent.yaml"), train_cfg)

    runner.learn(num_learning_iterations=train_cfg.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    if args.debug:
        args.offline = True
        args.num_envs = 1
    train(args)
