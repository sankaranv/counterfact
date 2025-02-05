from typing import Any, Callable, Dict, List, Union, Optional
import copy
import torch
import pyro.distributions as dist
import networkx as nx


class StructuralFunction:
    def __init__(
        self,
        function: Callable[
            [Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]], torch.Tensor
        ],
        parents: List[str],
        noise_dist: Optional[dist.Distribution] = None,
    ):
        self.function = function
        self.noise_dist = noise_dist
        self.parents = parents

    def sample(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Sample random noise and evaluate the structural function
        """
        self._check_parents(inputs)
        noise = self.noise_dist.sample() if self.noise_dist else None
        return self.function(inputs, noise)

    def evaluate(
        self,
        inputs: Dict[str, torch.Tensor],
        noise: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Evaluate the structural function given inputs and noise
        """
        self._check_parents(inputs)
        return self.function(inputs, noise)

    def _check_parents(self, inputs: Dict[str, torch.Tensor]):
        """
        Ensure all required parents are present in the inputs.
        """
        missing_parents = [p for p in self.parents if p not in inputs]
        if missing_parents:
            raise ValueError(f"Missing parent variables in input: {missing_parents}")

    def __call__(self, inputs, noise=None):
        return self.evaluate(inputs, noise)

    def __repr__(self):
        return f"StructuralFunction({self.function}, {self.parents}, {self.noise_dist})"


class StructuralCausalModel:
    def __init__(self, graph: Optional[nx.DiGraph] = None):
        """
        Initializes the SCM with a given causal graph.
        """
        self.variables: Dict[str, Any] = {}
        self.structural_functions: Dict[str, StructuralFunction] = {}
        self.interventions: Dict[
            str, Union[torch.Tensor, Callable[[], torch.Tensor]]
        ] = {}

        self.causal_graph = graph if graph else nx.DiGraph()
        self.original_graph = copy.deepcopy(self.causal_graph)
        self.original_functions: Dict[str, StructuralFunction] = {}
        self.topological_order = list(nx.topological_sort(self.causal_graph))
        self.formatted_var_names = {}

    def add_variable(
        self, var_name: str, var_type: str, support: List[Union[int, float]]
    ):
        """
        Adds a variable to the SCM.
        """
        if var_name in self.variables:
            raise ValueError(f"Variable {var_name} already exists.")
        self.validate_support(var_name, var_type, support)

        self.variables[var_name] = {"var_type": var_type, "support": support}
        self.causal_graph.add_node(var_name)
        self.original_graph.add_node(var_name)

        # Update topological ordering
        self.topological_order = list(nx.topological_sort(self.causal_graph))

    def add_variables(self, variables: Dict[str, Dict[str, Any]]):
        """
        Adds multiple variables to the SCM.
        """
        for name, details in variables.items():
            self.add_variable(name, details["var_type"], details["support"])

    def set_structural_function(
        self, var_name: str, structural_function: StructuralFunction
    ):
        """
        Set the structural function for a variable
        """
        if var_name not in self.variables:
            raise ValueError(f"Variable {var_name} not found.")
        self.structural_functions[var_name] = structural_function
        self.original_functions[var_name] = copy.deepcopy(structural_function)

        # Create edges in the causal graph from parents of the function to the variable
        for parent in structural_function.parents:
            self.causal_graph.add_edge(parent, var_name)

        # Update topological ordering
        self.topological_order = list(nx.topological_sort(self.causal_graph))

    def set_structural_functions(
        self, structural_functions: Dict[str, StructuralFunction]
    ):
        """
        Set the structural functions for a dict of variables
        """
        for var_name, structural_function in structural_functions.items():
            self.set_structural_function(var_name, structural_function)

    def reset(self):
        """
        Reset the SCM to its original state
        """
        self.interventions.clear()
        self.structural_functions = copy.deepcopy(self.original_functions)
        self.causal_graph = copy.deepcopy(self.original_graph)

    def freeze(self):
        """
        Save the current state of the SCM as the original state for all future resets
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
        self.validate_intervention(
            var_name, self.variables[var_name]["var_type"], value
        )

        # Remove all incoming edges to the intervened variable and store the intervened value
        self.interventions[var_name] = (
            torch.tensor(value) if not isinstance(value, torch.Tensor) else value
        )
        self.causal_graph.remove_edges_from(list(self.causal_graph.in_edges(var_name)))

        def constant_function(
            inputs: Dict[str, torch.Tensor],
            noise: Optional[Dict[str, torch.Tensor]] = None,
        ) -> torch.Tensor:

            value = self.interventions[var_name]
            return (
                value
                if isinstance(value, torch.Tensor)
                else torch.tensor(value, dtype=torch.float32)
            )

        # Structural function for the intervened variable is a constant function that returns the intervened value
        self.structural_functions[var_name] = StructuralFunction(
            function=constant_function, parents=[], noise_dist=None
        )

    def intervene(self, intervention):
        for var, value in intervention.items():
            self.do(var, value)

    def evaluate(
        self,
        var_name: str,
        inputs: Dict[str, torch.Tensor],
        noise: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Evaluate the value of a variable in the SCM by recursively evaluating the structural functions
        """

        if noise is None:
            noise = {}

        # Use intervention value if exists
        if var_name in self.interventions:
            value = self.interventions[var_name]
            return torch.tensor(value) if not isinstance(value, torch.Tensor) else value

        # Return input value if provided
        if var_name in inputs:
            return inputs[var_name]

        if var_name not in self.structural_functions:
            raise ValueError(f"Variable {var_name} not found.")

        parents = self.structural_functions[var_name].parents

        # Base case: variable has no exogenous parents
        if not parents:
            # Sample some random value for the noise if a noise distribution is provided
            if var_name not in noise:
                noise_dist = self.structural_functions[var_name].noise_dist
                noise[var_name] = (
                    noise_dist.sample() if noise_dist else torch.tensor(torch.nan)
                )

            return self.structural_functions[var_name].evaluate(inputs={}, noise=noise)

        # Recursive case: evaluate the structural function for the variable
        # Collect values of parents if they are not already given, where noise must be sampled if not given
        parent_values = {
            parent: (
                inputs[parent]
                if parent in inputs
                else self.evaluate(parent, inputs, noise)
            )
            for parent in parents
        }

        # Sample noise for the output variable if there is a noise distribution but a noise sample is not given
        if var_name not in noise:
            noise_dist = self.structural_functions[var_name].noise_dist
            noise[var_name] = (
                noise_dist.sample() if noise_dist else torch.tensor(torch.nan)
            )

        # Calculate the value of the given variable
        output = self.structural_functions[var_name].evaluate(inputs, noise)
        return output

    def sample(self, n_samples: int):
        """
        Sample from the SCM
        :param n_samples: Number of samples to generate
        :return: List of samples
        """

        samples = {var: torch.empty(n_samples) for var in self.variables}
        noise = {}

        for var_name in self.topological_order:

            # Use intervention value if present
            if var_name in self.interventions:
                value = (
                    torch.tensor(self.interventions[var_name])
                    if not isinstance(self.interventions[var_name], torch.Tensor)
                    else value
                )
                samples[var_name].fill_(value)
                continue

            # Gather parent values as tensors
            parents = self.structural_functions[var_name].parents
            parent_values = {parent: samples[parent] for parent in parents}
            # Sample noise if not already done
            if var_name not in noise:
                noise_dist = self.structural_functions[var_name].noise_dist
                noise[var_name] = (
                    noise_dist.sample((n_samples,))
                    if noise_dist
                    else torch.empty(n_samples)
                )

            # Evaluate the structural function
            output = self.structural_functions[var_name].evaluate(parent_values, noise)
            samples[var_name] = output

        return samples

    def get_state(
        self, noise: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:

        state = {}
        for var_name in self.topological_order:
            state[var_name] = self.evaluate(var_name, state, noise)
        return state

    def validate_support(self, name, var_type, support):
        if var_type == "bool":
            if not isinstance(support, list) or len(support) != 2:
                raise ValueError(
                    f"Support for boolean variable '{name}' must be list of two numbers."
                )
        elif var_type == "int":
            if (
                not isinstance(support, list)
                or len(support) != 2
                or not all(isinstance(x, int) for x in support)
            ):
                raise ValueError(
                    f"Support for integer variable '{name}' must be a list of two integers."
                )
        elif var_type == "float":
            if (
                not isinstance(support, list)
                or len(support) != 2
                or not all(isinstance(x, (int, float)) for x in support)
            ):
                raise ValueError(
                    f"Support for float variable '{name}' must be a list of two numbers."
                )
        elif var_type == "discrete":
            if not isinstance(support, list) or len(support) == 0:
                raise ValueError(
                    f"Support for discrete variable '{name}' must be a non-empty list of values."
                )
        else:
            raise ValueError(
                f"Unsupported variable type '{var_type}' for variable '{name}'."
            )

    def validate_intervention(self, name: str, var_type: str, value: Any):
        """Check if the intervened value is in the support of the variable.

        Args:
            name (str): _description_
            var_type (str): _description_
            value (Any): _description_
        """

        if var_type == "bool":
            if value not in [0, 1]:
                raise ValueError(
                    f"Value for boolean variable must be 0 or 1, {value} given."
                )
        elif var_type == "int":
            if int(value) != value:
                raise ValueError(
                    f"Value for integer variable must be an integer, {value} given"
                )
            if (
                value < self.variables[name].support[0]
                or value > self.variables[name].support[1]
            ):
                raise ValueError(
                    f"Value for integer variable must be in the range {self.variables[name].support}."
                )
        elif self.variables[name].var_type == "float":
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"Value for float variable must be a number, {value} given."
                )
            if (
                value < self.variables[name].support[0]
                or value > self.variables[name].support[1]
            ):
                raise ValueError(
                    f"Value for float variable must be in the range {self.variables[name].support}."
                )
        elif self.variables[name].var_type == "discrete":
            if value not in self.variables[name].support:
                raise ValueError(
                    f"Value for discrete variable must be one of {self.variables[name].support}."
                )
        else:
            raise ValueError(f"Unsupported variable type.")

    def __call__(
        self,
        var_name: str,
        inputs: Dict[str, torch.Tensor],
        noise: Optional[Dict[str, torch.Tensor]] = None,
    ):
        return self.evaluate(var_name, inputs, noise)
