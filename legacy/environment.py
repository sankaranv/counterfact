import pymc as pm
from pymc.model.transform.conditioning import do
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import gymnasium as gym


class Environment(gym.Env):
    def __init__(self, mode=None, n_samples_per_obs=1):

        self.n_samples_per_obs = n_samples_per_obs
        self.description = "Default AC Environment"
        self.mode = mode

        # Setup model
        self.model = pm.Model()
        self.build_model()
        self.vars = {key: i for i, key in enumerate(self.model.named_vars)}

        # Define the observation and action spaces to be the full set of real numbers
        # TODO - replace these with ranges
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_samples_per_obs, len(self.vars))
        )
        self.action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.vars),)
        )

    def do(self, vars_to_interventions):
        # Wrapper around the PyMC do operator
        return do(self.model, vars_to_interventions)

    def sample(self, **kwargs):
        # Sample from the model
        with self.model:
            trace = pm.sample(**kwargs)
        return trace

    def observe(self):
        # Sample from the model
        trace = self.sample(draws=self.n_samples_per_obs)
        sample = {
            key: trace.posterior.data_vars[key].data[0, 0]
            for key in trace.posterior.data_vars.keys()
        }
        return sample

    def step(self, action):
        # Perform the action and observe a new state
        self.model = self.do(action)
        observation = self.observe()
        # Return the next state, the reward, whether the episode is done, and additional info
        return observation, None, True, {}

    def reset(self):
        pass

    def render(self):
        # TODO - render the model with graphviz
        viz = pm.model_to_graphviz(self.model)
        viz.render("model", format="png", cleanup=True)

    def close(self):
        pass

    def build_model(self):
        pass
