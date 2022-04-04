from ray.tune.registry import register_env
######### Add new environment below #########

import gym
def build_env_with_agent_groups(env_class, config):
	env = env_class(config)
	grouping = {"group_1": list(range(config.get('num_agents',1)))}
	obs_space = gym.spaces.Tuple([env.observation_space]*config.get('num_agents',1))
	act_space = gym.spaces.Tuple([env.action_space]*config.get('num_agents',1))
	return env.with_agent_groups(grouping, obs_space=obs_space, act_space=act_space)

### Shepherd
from environments.shepherd.env import ShepherdEnv as Shepherd
register_env("Shepherd", lambda config: Shepherd(config))
register_env("Shepherd-Group", lambda config: build_env_with_agent_groups(Shepherd,config))