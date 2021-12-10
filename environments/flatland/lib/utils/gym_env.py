from collections import defaultdict
from typing import Dict, NamedTuple, Any, Optional

import gym
from gym import wrappers
from flatland.envs.rail_env import RailEnv,RailEnvActions

class StepOutput(NamedTuple):
    obs: Dict[int, Any]  # depends on observation builder
    reward: Dict[int, float]
    done: Dict[int, bool]
    info: Dict[int, Dict[str, Any]]


class FlatlandGymEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 10,
        'semantics.autoreset': True
    }

    def __init__(self,
                 rail_env: RailEnv,
                 observation_space: gym.spaces.Space,
                 render: bool = False,
                 regenerate_rail_on_reset: bool = True,
                 regenerate_schedule_on_reset: bool = True) -> None:
        super().__init__()
        self._agents_done = []
        self._agent_scores = defaultdict(float)
        self._agent_steps = defaultdict(int)
        self._agent_explanations = defaultdict(str)
        self._regenerate_rail_on_reset = regenerate_rail_on_reset
        self._regenerate_schedule_on_reset = regenerate_schedule_on_reset
        self.rail_env = rail_env
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = observation_space
        if render:
            self.rail_env.set_renderer(render)

    def step(self, action_dict: Dict[int, RailEnvActions]) -> StepOutput:
        d, r, o = None, None, None
        obs_or_done = False
        while not obs_or_done:
            # Perform env steps as long as there is no observation (for all agents) or all agents are done
            # The observation is `None` if an agent is done or malfunctioning.
            obs, rewards, dones, infos = self.rail_env.step(action_dict)

            d, r, o = dict(), dict(), dict()
            for agent, done in dones.items():
                if agent != '__all__' and not agent in obs:
                    continue  # skip agent if there is no observation

                # Use this if using a single policy for multiple agents
                # TODO find better way to handle this
                #if True or agent not in self._agents_done:
                if agent not in self._agents_done:
                    if agent != '__all__':
                        if done:
                            self._agents_done.append(agent)
                        o[agent] = obs[agent]
                        r[agent] = rewards[agent]
                        self._agent_scores[agent] += rewards[agent]
                        self._agent_steps[agent] += 1
                    d[agent] = dones[agent]
                if agent != '__all__':
                    self._agent_explanations[agent] = 'edit_me_plese_im_your_explanatory_label'

            action_dict = {}  # reset action dict for cases where we do multiple env steps
            obs_or_done = len(o) > 0 or d['__all__']  # step through env as long as there are no obs/all agents done

        assert all([x is not None for x in (d, r, o)])

        return StepOutput(obs=o, reward=r, done=d, info={agent: {
            'max_episode_steps': self.rail_env._max_episode_steps,
            'num_agents': self.rail_env.get_num_agents(),
            'agent_done': d[agent] and agent not in self.rail_env.active_agents,
            'agent_score': self._agent_scores[agent],
            'agent_step': self._agent_steps[agent],
            'explanation': self._agent_explanations[agent],
        } for agent in o.keys()})

    def reset(self, random_seed: Optional[int] = None) -> Dict[int, Any]:
        self._agents_done = []
        self._agent_scores = defaultdict(float)
        self._agent_steps = defaultdict(int)
        self._agent_explanations = defaultdict(str)
        obs, infos = self.rail_env.reset(regenerate_rail=self._regenerate_rail_on_reset,
                                         regenerate_schedule=self._regenerate_schedule_on_reset,
                                         random_seed=random_seed)
        return {k: o for k, o in obs.items() if not k == '__all__'}

    def render(self, mode='human'):
        return self.rail_env.render(mode)

    def close(self):
        self.rail_env.close()
