from ray.rllib import MultiAgentEnv

from game import ShepherdGame
from observer import ShepherdObserver


class ShepherdEnv(MultiAgentEnv):
    def __init__(self, num_dogs, num_sheep):
        self.num_dogs = num_dogs
        self.num_sheep = num_sheep
        self.game = ShepherdGame(self.num_dogs, self.num_sheep, render=True, save_frames=False)
        self.observer = ShepherdObserver(self.game)

    def reset(self):
        self.game = ShepherdGame(self.num_dogs, self.num_sheep)
        self.observer = ShepherdObserver(self.game)

    def run(self):
        while True:
            self.step({})

    def step(self, action_dict):
        fake_action_dict = {i: i for i in range(self.num_dogs)}
        self.game.step(fake_action_dict)
        self.observer.update()
        for agent in self.game.dogs:
            print(f"Explanations for agent {agent.pid}: {self.observer.explain(agent)}")

env = ShepherdEnv(10, 50)
env.run()
