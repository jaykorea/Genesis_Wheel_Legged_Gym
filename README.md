# [🛞](https://github.com/user-attachments/assets/01ebe3a4-c2dd-4215-8128-882aee685234) Wheeled Legged GYM with Genesis AI

[![Genesis](https://img.shields.io/badge/genesis_ai-blue)](https://genesis-world.readthedocs.io/en/latest/user_guide/index.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-bsd-yellow.svg)](https://opensource.org/license/mit)

## **✨ New Features - Updated 🚀**
✔️ **Compatibility**: Fully adaptable with  leg / wheel robots.  
✔️ **Flamingo rev.1.5.1**: Latest version of Flamingo added.  
✔️ **Flamingo Light v1**: Flamingo Light version added.  
✔️ **Stack Environment**: Observations can be stacked with arguments(cfg).  
✔️ **RSL_RL**: latest version of rsl-rl(2.2.4) lib adopted.  

## 🌟 Features
- **🧠 Genesis AI Integration**
  Leverages the Genesis simulation framework, enabling high-speed, GPU-accelerated, and differentiable physics environments for advanced robotic learning.
- **🦿 Wheeled-Legged Hybrid Robot Support**
  Seamlessly supports both legged and wheeled locomotion. Includes configurations for Flamingo and Flamingo Light platforms.
- **⚙️ Modular Environment Structure**
  Built upon legged_gym and follows a modular architecture for easy customization of observations, rewards, actions, and terrains.
-	**🧪 Sim-to-Real Transfer Friendly**
  Built with domain randomization, actuator noise modeling, and observation smoothing techniques for zero-shot transfer to real hardware.
- **📚 RSL-RL Integration (v2.2.4)**
  Fully compatible with the latest rsl-rl release, ensuring easy use of PPO and other RL algorithms.

---
## 🛠 Installation
1. Create Conda Env (python >= 3.10 recommend)
  ```
conda create -n genesis-wlr python=3.10
  ```
2. Install torch
  ```
  https://pytorch.org/get-started/locally/
  ```
3. Clone repo and install requirements
  ```
  git clone https://github.com/jaykorea/Genesis_Wheeled_Legged_Gym.git
  pip install -e .
  ```
4. Install Latest Genesis AI (Direct install from git recommended)
  ```
  git clone https://github.com/Genesis-Embodied-AI/Genesis.git
  cd Genesis
  pip install -e ".[dev]"
  ```
---

## 👋 Usage

### 🚀 Quick Start

By default, the task is set to `go2`(in `utils/helper.py`), we can run a training session with the following command:

```bash
cd wheeled_legged_gym/scripts
python train.py --task flamingo --num_envs 4096 --headless
```

After the training is done, run `play.py` to visualize the trained model:  
By default, if 'load_run' argument is not provided, it will load the recent model path.
```bash
cd wheeled_legged_gym/scripts
python play.py --task flamingo --num_envs 1 
```

## 🧪 Test Results

### Flamingo Sim2Real
### Flamingo Light Sim2Real

## 🙏 Acknowledgements

- [Genesis](https://github.com/Genesis-Embodied-AI/Genesis/tree/main)
- [Genesis-backflip](https://github.com/ziyanx02/Genesis-backflip)
- [legged_gym](https://github.com/leggedrobotics/legged_gym)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl)
- [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)
- [genesis legged gym](https://github.com/lupinjia/genesis_lr)

## TODO
- [ ] Add env origin handler
- [ ] Add ISIM
