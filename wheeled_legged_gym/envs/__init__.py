# go2
from wheeled_legged_gym.envs.go2.go2 import GO2
from wheeled_legged_gym.envs.go2.go2_config import GO2Cfg, GO2CfgPPO
# go2_rough
from wheeled_legged_gym.envs.go2.go2_rough.go2_rough_config import GO2RoughCfg, GO2RoughCfgPPO
# flamingo_light
from wheeled_legged_gym.envs.flamingo_light.flamingo_light import FlamingoLight
from wheeled_legged_gym.envs.flamingo_light.flamingo_light_config import FlamingoLightCfg, FlamingoLightCfgPPO
# flamingo
from wheeled_legged_gym.envs.flamingo.flamingo import Flamingo
from wheeled_legged_gym.envs.flamingo.flamingo_config import FlamingoCfg, FlamingoCfgPPO
# common
from wheeled_legged_gym.utils.task_registry import task_registry

task_registry.register( "go2", GO2, GO2Cfg(), GO2CfgPPO())
task_registry.register( "go2_rough", GO2, GO2RoughCfg(), GO2RoughCfgPPO())
task_registry.register( "flamingo_light", FlamingoLight, FlamingoLightCfg(), FlamingoLightCfgPPO())
task_registry.register( "flamingo", Flamingo, FlamingoCfg(), FlamingoCfgPPO())