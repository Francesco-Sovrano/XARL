from ray.rllib.env.multi_agent_env import MultiAgentEnv, make_multi_agent
import gym
import numpy as np

from environments.shepherd.game import ShepherdGame
from environments.shepherd.observer import ShepherdObserver


class ShepherdEnv(MultiAgentEnv):
    def __init__(self, config):
        num_dogs = config.get('num_dogs',5)
        num_sheep = config.get('num_sheep',50)
        self.num_dogs = num_dogs
        self.num_sheep = num_sheep
        self.render = False
        self.save_frames = False
        self.game = ShepherdGame(self.num_dogs, self.num_sheep, render=self.render, save_frames=self.save_frames)
        self.observer = ShepherdObserver(self.game)

    def reset(self):
        self.game = ShepherdGame(self.num_dogs, self.num_sheep, render=self.render, save_frames=self.save_frames)
        self.observer = ShepherdObserver(self.game)
        return self.observation_space.sample()

    def run(self):
        fake_action_dict = {i: i for i in range(self.num_dogs)}
        while True:
            observations, reward, done, explanations = self.step(fake_action_dict)
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
        self.game.step(action_dict)
        observations, reward, done = self.observer.update()
        explanations = {}
        for agent in self.game.dogs:
            explanations[agent] = self.observer.explain(agent)
        return observations, reward, done, explanations

if __name__ == '__main__':
    env = ShepherdEnv({
        'num_dogs': 10,
        'num_sheep': 50,
    })
    print(env.observation_space.sample())
    env.run()
