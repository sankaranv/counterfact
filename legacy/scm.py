from actual_cause.dag import CausalDAGEnv
import torch
import pyro


class StructuralCausalModel:
    def __init__(self):
        self.functions = {}
        self.params = {}
        self.ranges = {}
        self.observed_nodes = set()
        self.latent_nodes = set()
        self.node_names = []

    def create_variable(self, name, var_type, is_latent=False):
        """
        Create a variable in the SCM
        :param name: name of the variable
        :param var_type: type of the variable
        :param is_latent: whether the variable is latent
        :return: None
        """

        # Create pyro parameters for each variable
        if var_type == "bool":
            self.params[name] = pyro.param(name, torch.tensor(0.5))

    def sample(self, n_samples=1):
        pass
