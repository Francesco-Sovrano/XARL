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
        self.observer = ShepherdObserver(self.game)

    def reset(self):
        self.game = ShepherdGame(self.num_dogs, self.num_sheep, render=self.render, save_frames=self.save_frames)
        self.observer = ShepherdObserver(self.game)

    def run(self):
        while True:
            observations, reward, done, explanations = self.step({})
            if done:
                exit(999)

    @property
    def observation_space(self) -> gym.spaces.Space:
        return gym.spaces.Dict({
            'agent_pos': gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
            'pen_pos': gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
            'local_view': gym.spaces.Box(low=0, high=5, shape=(61,61), dtype=np.int8), #TODO: Send to CNN
        })

    @property
    def action_space(self):
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def step(self, action_dict):
        fake_action_dict = {i: i for i in range(self.num_dogs)}
        self.game.step(fake_action_dict)
        observations, reward, done = self.observer.update()
        explanations = {}
        for agent in self.game.dogs:
            explanations[agent] = self.observer.explain(agent)
        return observations, reward, done, explanations

env = ShepherdEnv(10, 50)
# print(env.observation_space.sample())
env.run()
