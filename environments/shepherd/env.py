import random

from ray.rllib.env.multi_agent_env import MultiAgentEnv, make_multi_agent
import gym
import numpy as np
from environments.shepherd.game import ShepherdGame
from environments.shepherd.observer import ShepherdObserver


class ShepherdEnv(MultiAgentEnv):
    def __init__(self, config):
        self.num_dogs = config.get('num_dogs',5)
        self.num_sheep = config.get('num_sheep',50)
        self.dog_sense_radius = config.get('dog_sense_radius',125)
        self.map_sparsity = config.get('map_sparsity',5)
        self.render = False
        self.save_frames = False
        self.game = ShepherdGame(self.num_dogs, self.num_sheep, render=self.render, save_frames=self.save_frames, dog_sense_radius=self.dog_sense_radius, map_sparsity=self.map_sparsity)
        self.observer = ShepherdObserver(self.game)
        self._agent_ids = set(range(self.num_dogs))

    def reset(self):
        self.game = ShepherdGame(self.num_dogs, self.num_sheep, render=self.render, save_frames=self.save_frames, dog_sense_radius=self.dog_sense_radius, map_sparsity=self.map_sparsity)
        self.observer = ShepherdObserver(self.game)
        # return {
        #     i: self.observation_space.sample() 
        #     for i in range(self.num_dogs)
        # }
        return self.observer.update()[0]

    def run(self):
        self.reset()
        fake_action_dict = {i: np.array([1.0, 1.0]) for i in range(self.num_dogs)}
        while True:
            observations, reward, done, explanations = self.step(fake_action_dict)
            sample_obs = env.observation_space.sample()
            print(f"frame: {self.game.frame_count}\nexplanations: {explanations}")
            # for k in observations.keys():
            #     if observations[k]['local_view'].shape != (61, 61):
            #         print("Wrong shape")
            if done['__all__']:
                exit(999)


    @property
    def observation_space(self) -> gym.spaces.Space:
        view_dim = int(self.game.dog_sense_radius*2)
        return gym.spaces.Dict({
            "cnn": gym.spaces.Dict({
                "grid": gym.spaces.Box(low=0, high=1, shape=(view_dim,view_dim,5), dtype=np.uint8)
            }),
        })
        # return gym.spaces.Box(low=0, high=1, shape=(view_dim*view_dim*5,), dtype=np.uint8)

    @property
    def action_space(self):
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def step(self, action_dict):
        #print(action_dict)
        self.game.step(action_dict)
        observations, reward, done = self.observer.update()
        info = {
            k: {'explanation': self.observer.explain(k)}
            for k in observations.keys()
        }
        return observations, reward, done, info

if __name__ == '__main__':
    # seed = 1
    # random.seed(seed)
    env = ShepherdEnv({
        'num_dogs': 10,
        'num_sheep': 50,
    })
    # print(env.observation_space.sample())
    env.run()


