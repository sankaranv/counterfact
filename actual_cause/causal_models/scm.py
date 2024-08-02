from typing import Any, Callable, Dict, List, Union
from actual_cause.causal_models.variables import ExogenousNoise, Variable
from actual_cause.causal_models.graphs import CausalDAG
import copy
import networkx as nx


class StructuralFunction:
    def __init__(
        self,
        function: Callable[..., float],
        parents: List[str],
        noise_dist: ExogenousNoise = None,
    ):
        self.function = function
        self.noise_dist = noise_dist
        self.parents = parents

    def sample(self, inputs) -> float:
        """
        Sample random noise and evaluate the structural function
        :param inputs
        :return:
        """
        for parent in self.parents:
            if parent not in inputs:
                raise ValueError(f"Parent {parent} not found in input.")
        if self.noise_dist is not None:
            noise = self.noise_dist.sample()
        else:
            noise = None
        return self.function(inputs, noise)

    def evaluate(self, inputs, noise=None) -> float:
        """
        Evaluate the structural function given inputs and noise
        :param inputs:
        :param noise:
        :return:
        """
        for parent in self.parents:
            if parent not in inputs:
                raise ValueError(f"Parent {parent} not found in input.")
        #
        return self.function(inputs, noise)

    def __call__(self, inputs, noise=None):
        return self.evaluate(inputs, noise)

    def __repr__(self):
        return f"StructuralFunction({self.function}, {self.parents}, {self.noise_dist})"


class StructuralCausalModel:
    def __init__(self, graph: CausalDAG = None):
        self.variables: Dict[str, Variable] = {}
        self.structural_functions: Dict[str, StructuralFunction] = {}
        self.interventions: Dict[str, Union[float, Callable[[], float]]] = {}
        if graph is None:
            self.causal_graph = CausalDAG()
        else:
            self.causal_graph = graph
        self.original_graph = copy.deepcopy(self.causal_graph)
        self.original_functions: Dict[str, StructuralFunction] = {}
        self.topological_order = None
        self.formatted_var_names = {}

    def add_variable(self, variable: Variable):
        """
        Add a variable to the SCM
        :param variable: Variable object
        :return: None
        """
        if variable.name in self.variables:
            raise ValueError(f"Variable {variable.name} already exists")

        # Add variable to the model and causal graph
        self.variables[variable.name] = variable
        self.causal_graph.add_node(variable.name)
        self.original_graph.add_node(variable.name)

        # Update topological ordering
        self.topological_order = list(nx.topological_sort(self.causal_graph))

    def add_variables(self, variables: List[Variable]):
        """
        Add a list of variables to the SCM
        :param variables: List of Variable objects
        :return: None
        """
        for variable in variables:
            self.add_variable(variable)

    def set_structural_function(
        self, var_name: str, structural_function: StructuralFunction
    ):
        """
        Set the structural function for a variable
        :param var_name: Name of the variable
        :param structural_function: StructuralFunction object
        :return:
        """
        if var_name not in self.variables:
            raise ValueError(f"Variable {var_name} not found.")
        self.structural_functions[var_name] = structural_function
        self.original_functions[var_name] = copy.deepcopy(structural_function)

        # Create edges in the causal graph from parents of the function to the variable
        for parent in structural_function.parents:
            self.add_edge(parent, var_name)

        # Update topological ordering
        self.topological_order = list(nx.topological_sort(self.causal_graph))

    def set_structural_functions(
        self, structural_functions: Dict[str, StructuralFunction]
    ):
        """
        Set the structural functions for a dict of variables
        :param structural_functions: Dict of variable names to StructuralFunction objects
        :return: None
        """
        for var_name, structural_function in structural_functions.items():
            self.set_structural_function(var_name, structural_function)

    def add_edge(self, from_node: str, to_node: str):
        self.causal_graph.add_edge(from_node, to_node)
        self.original_graph.add_edge(from_node, to_node)

    def reset(self):
        """
        Reset the SCM to its original state
        :return: None
        """
        self.interventions.clear()
        self.structural_functions = copy.deepcopy(self.original_functions)
        self.causal_graph = copy.deepcopy(self.original_graph)

    def freeze(self):
        """
        Save the current state of the SCM as the original state for all future resets
        :return: None
        """
        self.original_graph = copy.deepcopy(self.causal_graph)
        self.original_functions = copy.deepcopy(self.structural_functions)

    def do(self, var_name: str, value: Any):
        """
        Perform an intervention on a variable
        :param var_name: Name of the variable to intervene on
        :param value: Intervened value
        :return: None
        """
        # Check if intervened value is in the support of the variable
        if var_name not in self.variables:
            raise ValueError(f"Variable {var_name} not found.")
        if self.variables[var_name].var_type == "bool":
            if value not in [0, 1]:
                raise ValueError(
                    f"Value for boolean variable must be 0 or 1, {value} given."
                )
        elif self.variables[var_name].var_type == "int":
            if int(value) != value:
                raise ValueError(
                    f"Value for integer variable must be an integer, {value} given"
                )
            if (
                value < self.variables[var_name].support[0]
                or value > self.variables[var_name].support[1]
            ):
                raise ValueError(
                    f"Value for integer variable must be in the range {self.variables[var_name].support}."
                )
        elif self.variables[var_name].var_type == "float":
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"Value for float variable must be a number, {value} given."
                )
            if (
                value < self.variables[var_name].support[0]
                or value > self.variables[var_name].support[1]
            ):
                raise ValueError(
                    f"Value for float variable must be in the range {self.variables[var_name].support}."
                )
        elif self.variables[var_name].var_type == "discrete":
            if value not in self.variables[var_name].support:
                raise ValueError(
                    f"Value for discrete variable must be one of {self.variables[var_name].support}."
                )
        else:
            raise ValueError(f"Unsupported variable type.")

        # Remove all incoming edges to the intervened variable and store the intervened value
        self.interventions[var_name] = value
        self.causal_graph.remove_edges_from(list(self.causal_graph.in_edges(var_name)))

        def constant_function(inputs, noise):
            return value

        # Structural function for the intervened variable is a constant function that returns the intervened value
        self.structural_functions[var_name] = StructuralFunction(
            function=constant_function, noise_dist=None, parents=[]
        )

    def intervene(self, intervention):
        for var, value in intervention.items():
            self.do(var, value)

    def evaluate(self, var_name: str, inputs: dict, noise: dict = None) -> float:
        """
        Evaluate the value of a variable in the SCM by recursively evaluating the structural functions
        :param var_name:
        :param inputs: Dict of inputs to the function
        :param noise: Dict of noise values for each variable
        :return:
        """

        if noise is None:
            noise = {}
        if var_name in self.interventions:
            return self.interventions[var_name]
        if var_name in inputs:
            return inputs[var_name]
        if var_name not in self.structural_functions:
            raise ValueError(f"Variable {var_name} not found.")

        # Base case: variable has no exogenous parents
        if not self.structural_functions[var_name].parents:
            if var_name not in noise:
                # Sample some random value for the noise if a noise distribution is provided
                if self.structural_functions[var_name].noise_dist is not None:
                    noise[var_name] = self.structural_functions[
                        var_name
                    ].noise_dist.sample()
                else:
                    noise[var_name] = None

            return self.structural_functions[var_name].evaluate(inputs={}, noise=noise)

        # Recursive case: evaluate the structural function for the variable
        # Collect values of parents if they are not already given, where noise must be sampled if not given
        for parent in self.structural_functions[var_name].parents:
            if parent not in inputs:
                # If the parent was intervened on, use the intervened value
                if parent in self.interventions:
                    inputs[parent] = self.interventions[parent]
                else:
                    # Sample noise for the parent variable if there is a noise distribution but a noise sample is not given
                    if parent not in noise:
                        if self.structural_functions[parent].noise_dist is not None:
                            noise[parent] = self.structural_functions[
                                parent
                            ].noise_dist.sample()
                        else:
                            noise[parent] = None

                    # Calculate the value of the parent variable
                    inputs[parent] = self.evaluate(parent, inputs, noise)

        # Sample noise for the output variable if there is a noise distribution but a noise sample is not given
        if var_name not in noise:
            if self.structural_functions[var_name].noise_dist is not None:
                noise[var_name] = self.structural_functions[
                    var_name
                ].noise_dist.sample()
            else:
                noise[var_name] = None

        # Calculate the value of the given variable
        output = self.structural_functions[var_name].evaluate(inputs, noise)
        return output

    def sample(self, n_samples: int):
        """
        Sample from the SCM
        :param n_samples: Number of samples to generate
        :return: List of samples
        """
        samples = []
        nodes = list(nx.topological_sort(self.causal_graph))
        for _ in range(n_samples):
            sample = {}
            for var_name in nodes:
                if var_name not in sample:
                    sample[var_name] = self.evaluate(var_name, sample)
            samples.append(sample)
        return samples

    def get_state(self, noise: dict = None):
        nodes = list(nx.topological_sort(self.causal_graph))
        state = {}
        for var_name in nodes:
            if var_name not in state:
                state[var_name] = self.evaluate(var_name, state, noise)
        return state

    def __call__(self, var_name: str, **kwargs: Any) -> float:
        return self.evaluate(var_name, **kwargs)
