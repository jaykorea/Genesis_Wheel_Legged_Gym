# [🛞](https://github.com/user-attachments/assets/01ebe3a4-c2dd-4215-8128-882aee685234) Wheeled Legged GYM with Genesis AI

[![Genesis](https://github.com/user-attachments/assets/01ebe3a4-c2dd-4215-8128-882aee685234)](https://genesis-world.readthedocs.io/en/latest/user_guide/index.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

## **✨ New Features - Updated 🚀**
✔️ **Flamingo rev.1.5.1: Latest version of Flamingo added.  
✔️ **Flamingo Light v1**: Flamingo Light version added.  
✔️ **Stack Environment**: Observations can be stacked with arguments(cfg).
✔️ **RSL_RL**: latest version of rsl-rl(2.2.4) lib adopted

---
## 🏃‍♂️ Install
1. Create Conda Env
python >= 3.10 recommend
```
conda create -n genesis-wlr python=3.10

```
1. Install Latest Genesis AI (Direct install from git recommended)
```
git clone https://github.com/Genesis-Embodied-AI/Genesis.git
cd Genesis
pip install -e ".[dev]"
```
---


## 🌟 Features

- **Totally based on [legged_gym](https://github.com/leggedrobotics/legged_gym)**
  
  It's easy to use for those who are familiar with legged_gym and rsl_rl

- **Faster and Smaller**
  
  For a go2 walking on the plane task with 4096 envs, the training speed in Genesis is approximately **1.3x** compared to [Isaac Gym](https://developer.nvidia.com/isaac-gym), while the graphics memory usage is roughly **1/2** compared to IsaacGym.

  With this smaller memory usage, it's possible to **run more parallel environments**, which can further improve the training speed.

## 🧪 Test Results

For tests conducted on Genesis, please refer to [tests.md](./test_resources/tests.md)

## 🛠 Installation

1. Create a new python virtual env with python>=3.9
2. Install [PyTorch](https://pytorch.org/)
3. Install Genesis following the instructions in the [Genesis repo](https://github.com/Genesis-Embodied-AI/Genesis)
4. Install rsl_rl and tensorboard
   ```bash
   # Install rsl_rl.
   git clone https://github.com/leggedrobotics/rsl_rl
   cd rsl_rl && git checkout v1.0.2 && pip install -e . --use-pep517

   # Install tensorboard.
   pip install tensorboard
   ```
5. Install genesis_lr
   ```bash
   git clone https://github.com/lupinjia/genesis_lr
   cd genesis_lr
   pip install -e .
   ```

## 👋 Usage

### 🚀 Quick Start

By default, the task is set to `go2`(in `utils/helper.py`), we can run a training session with the following command:

```bash
cd legged_gym/scripts
python train.py --headless # run training without rendering
```

After the training is done, paste the `run_name` under `logs/go2` to `load_run` in `go2_config.py`: 

![](./test_resources/paste_load_run.png)

Then, run `play.py` to visualize the trained model:

![](./test_resources/go2_flat_play.gif)

### 📖 Instructions

For more detailed instructions, please refer to the [wiki page](https://github.com/lupinjia/genesis_lr/wiki)

## 🖼️ Gallery

| Go2 | Bipedal Walker |
|--- | --- |
| ![](./test_resources/go2_flat_play.gif) | ![](./test_resources/bipedal_walker_flat.gif) |

## 🙏 Acknowledgements

- [Genesis](https://github.com/Genesis-Embodied-AI/Genesis/tree/main)
- [Genesis-backflip](https://github.com/ziyanx02/Genesis-backflip)
- [legged_gym](https://github.com/leggedrobotics/legged_gym)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl)
- [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)

## TODO

- [x] Add domain randomization
- [x] Verify the trained model on real robots.
- [x] Add Heightfield support
- [x] Add meausre_heights support
- [ ] Add go2 deploy demos and instructions (vanilla and explicit estimator)
- [ ] Add teacher-student implementation
