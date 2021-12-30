from ray.rllib.env.multi_agent_env import MultiAgentEnv, make_multi_agent
import gym
import numpy as np

from game import ShepherdGame
from observer import ShepherdObserver


class ShepherdEnv(MultiAgentEnv):
    def __init__(self, num_dogs, num_sheep):
        self.num_dogs = num_dogs
        self.num_sheep = num_sheep
        self.render = True
        self.save_frames = False
        self.game = ShepherdGame(self.num_dogs, self.num_sheep, render=self.render, save_frames=self.save_frames)
        self.max_observed_neighbours = 20
        self.observer = ShepherdObserver(self.game, self.max_observed_neighbours)

    def reset(self):
        self.game = ShepherdGame(self.num_dogs, self.num_sheep, render=self.render, save_frames=self.save_frames)
        self.observer = ShepherdObserver(self.game, self.max_observed_neighbours)

    def run(self):
        while True:
            self.step({})

    @property
    def observation_space(self) -> gym.spaces.Space:
        return gym.spaces.Dict({
            'agent_pos': gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
            'pen_pos': gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
            'neighbours': gym.spaces.Dict({
                'types': gym.spaces.Box(low=0, high=2, shape=(self.max_observed_neighbours,), dtype=np.int8),
                'pos': gym.spaces.Box(low=0.0, high=1.0, shape=(self.max_observed_neighbours, 2), dtype=np.float32)
            })
        })

    @property
    def action_space(self):
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def step(self, action_dict):
        fake_action_dict = {i: i for i in range(self.num_dogs)}
        self.game.step(fake_action_dict)
        self.observer.update()
        for agent in self.game.dogs:
            # print(f"Explanations for agent {agent.pid}: {self.observer.explain(agent)}")
            pass

env = ShepherdEnv(20, 50)
# print(env.observation_space.sample())
env.run()
