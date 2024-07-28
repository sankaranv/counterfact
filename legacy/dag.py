import gymnasium as gym
import networkx as nx
import graphviz
from copy import deepcopy
from ranges import *
import numpy as np
import torch

class CausalDAGEnv(gym.Env):
    def __init__(self):
        self.dag = nx.DiGraph()
        self.ranges = {}
        self.observed_nodes = set()
        self.latent_nodes = set()
        self.node_names = []
        self.observation_space = gym.spaces.Dict()
        self.action_space = gym.spaces.Dict()

    def _do(self, node, value):
        """
        Apply in-place intervention on the node to the given value
        :param node:
        :return: None
        """

        assert node in self.observed_nodes, f"Variable {node} is not observed, cannot intervene"

        # Check if the value is in the range
        if value not in self.ranges[node]:
            raise ValueError(f"Value {value} is not in [{self.ranges[node]}] range for {node}")

        # Set the value of the node
        self.dag.nodes[node]['value'] = value

        # Remove incoming edges to node
        for parent in self.dag.predecessors(node):
            self.dag.remove_edge(parent, node)

    def do(self, node, value):
        """
        Apply intervention on the node to the given value and return the new model
        :param node:
        :return: None
        """
        intervened_model = deepcopy(self)
        intervened_model._do(node, value)
        return intervened_model

    def _intervene_state(self, values, null_value=None):
        """
        Apply intervention on the nodes to the given values
        :param values: can be a dictionary, numpy array or torch tensor, where null_value indicates no intervention
        :return: None
        """
        for node, value in values.items():
            if value != null_value:
                self._do(node, value)

    def intervene_state(self, values, null_value=None):
        """
        Apply intervention on the nodes to the given values and return the new model
        :param values: can be a dictionary, numpy array or torch tensor, where null_value indicates no intervention
        :return: None
        """
        intervened_model = deepcopy(self)
        intervened_model._intervene_state(values, null_value=null_value)
        return intervened_model

    def add_variable(self, name, var_type, value=None, is_latent=False):
        """
        Add a variable to the DAG
        :param name: name of the variable
        :param range: range of the variable
        :param is_latent: whether the variable is latent or not
        :return: None
        """
        observed = not is_latent
        self.dag.add_node(name, observed=observed, value=value)
        self.ranges[name] = var_type
        if not is_latent:
            self.observed_nodes.add(name)
        else:
            self.latent_nodes.add(name)
        self.node_names = list(self.observed_nodes)

    def build_dag(self, observed_nodes, ranges, latent_nodes=None, edges=None):
        """
        Build the DAG from the given nodes and ranges
        :param observed_nodes:
        :param latent_nodes:
        :param ranges:
        :return:
        """
        self.observed_nodes = set(observed_nodes)
        self.latent_nodes = set(latent_nodes)
        self.ranges = ranges
        self.dag = nx.DiGraph()

        # Add nodes
        for node in observed_nodes:
            self.dag.add_node(node, observed=True, value=None)

        for node in latent_nodes:
            self.dag.add_node(node, observed=False, value=None)

        # Add edges
        for node, parents in edges:
            for parent in parents:
                self.dag.add_edge(parent, node)

        # Add ranges
        for node, variable_range in ranges.items():
            if (not isinstance(range, ContinuousRange)
                    and not isinstance(variable_range, DiscreteRange)
                    and not isinstance(variable_range, IntegerRange)
                    and not isinstance(variable_range, BooleanRange)):
                raise ValueError(f"Range for {node} should be ContinuousRange, DiscreteRange, IntegerRange or BooleanRange, got {type(variable_range)}")
            self.ranges[node] = variable_range

        # Save names of nodes as a vector, which is used when returning states
        self.node_names = list(observed_nodes)

        # Set observation space, which is a dictionary of spaces
        # Each observation is a (node, value) pair
        spaces = {}
        for node in observed_nodes:
            if isinstance(ranges[node], ContinuousRange):
                spaces[node] = gym.spaces.Box(low=ranges[node].low, high=ranges[node].high, shape=(1,))
            elif isinstance(ranges[node], DiscreteRange):
                spaces[node] = gym.spaces.Discrete(n=len(ranges[node].values))
            elif isinstance(ranges[node], IntegerRange):
                spaces[node] = gym.spaces.Discrete(n=ranges[node].high - ranges[node].low + 1)
            elif isinstance(ranges[node], BooleanRange):
                spaces[node] = gym.spaces.Discrete(n=2)
        self.observation_space = gym.spaces.Dict(spaces)

        # Action space and observation space are identical for causal DAGs
        self.action_space = gym.spaces.Dict(spaces)

    def observe(self, return_type='dict'):
        """
        Observe the values of the nodes in the DAG
        :param return_type: 'dict' or 'numpy' or 'torch'
        :return: values of all observed nodes in the DAG
        """

        if return_type == 'dict':
            return {node: self.dag.nodes[node]['value'] for node in self.observed_nodes}
        elif return_type == 'numpy':
            return np.array([self.dag.nodes[node]['value'] for node in self.observed_nodes])
        elif return_type == 'torch':
            return torch.tensor([self.dag.nodes[node]['value'] for node in self.observed_nodes])
        else:
            raise ValueError(f"return_type should be 'dict', 'numpy' or 'torch', got {return_type}")

    def init(self, values):
        """
        Initialize the values of the observed nodes in the DAG
        :param values: values for observed nodes, can be a dictionary, numpy array or torch tensor
        :return: None
        """

        if isinstance(values, dict):
            for node, value in values.items():
                self.dag.nodes[node]['value'] = value
        elif isinstance(values, np.ndarray) or isinstance(values, torch.Tensor):
            for i, node in enumerate(self.observed_nodes):
                self.dag.nodes[node]['value'] = values[i]
        else:
            raise ValueError(f"values should be a dictionary, numpy array or torch tensor, got {type(values)}")

    def set_context(self, context):
        """
        Set the values for latent nodes in the DAG
        :param context: values for latent nodes, can be a dictionary, numpy array or torch tensor
        :return: None
        """

        if isinstance(context, dict):
            for node, value in context.items():
                if node not in self.latent_nodes:
                    raise ValueError(f"Node {node} is not a latent node")
                self.dag.nodes[node]['value'] = value
        elif isinstance(context, np.ndarray) or isinstance(context, torch.Tensor):
            for i, node in enumerate(self.latent_nodes):
                if node not in self.latent_nodes:
                    raise ValueError(f"Node {node} is not a latent node")
                self.dag.nodes[node]['value'] = context[i]
        else:
            raise ValueError(f"Context should be a dictionary, numpy array or torch tensor, got {type(context)}")

    def sample(self, n_samples=1):
        """
        Sample from the DAG
        :param n_samples: number of samples
        :return: samples
        """
        raise NotImplementedError

    def step(self, action, return_type='dict'):

        """
        A step in the environment is applying an intervention
        :param node:
        :return: None
        """

        node, value = list(action.items())[0]
        self._do(node, value)
        # Get state of all nodes in the DAG
        state = self.observe(return_type=return_type)
        return state, 0, False, {}

    def reset(self):
        """
        Reset the environment
        :return: None
        """
        for node in self.observed_nodes:
            self.dag.nodes[node]['value'] = None

    def render(self, mode='human'):
        """
        Render the DAG
        :param mode:
        :return:
        """

        dot = graphviz.Digraph()

        # Make observed nodes white and latent nodes grey
        for node in self.dag.nodes:
            if self.dag.nodes[node]['observed']:
                dot.node(node, label=f"{node} = {self.dag.nodes[node]['value']}", style='filled', fillcolor='white')
            else:
                dot.node(node, label=f"{node} = {self.dag.nodes[node]['value']}", style='filled', fillcolor='grey')

        # Render edges
        for edge in self.dag.edges:
            dot.edge(edge[0], edge[1])

        # TODO - give different render modes and save options
        dot.render('causal_dag', format='png', cleanup=True)
