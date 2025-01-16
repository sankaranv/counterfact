import pandas as pd
import itertools
from typing import List
from actual_cause.causal_models.variables import Variable, ExogenousNoise
from actual_cause.causal_models.scm import StructuralCausalModel, StructuralFunction
from actual_cause.definitions import ACDefinition
from actual_cause.inference import *


class ACSolver:

    def __init__(self, env: StructuralCausalModel, ac_defn: ACDefinition):
        self.env = env
        self.ac_defn = ac_defn

    def solve_all_states(
        self, env: StructuralCausalModel, ac_defn: ACDefinition, outcome_vars: List[str]
    ):
        """
        Find all actual causes in all reachable states for a given outcome variable
        :param env: StructuralCausalModel
        :param ac_defn: ACDefinition
        :param outcome_vars: List of outcome variable names
        :return: dataframe containing states, outcomes and actual causes
        """

        # Prepare table
        state_vars = sorted(
            list(env.variables.keys()), key=lambda x: env.topological_order.index(x)
        )
        outcome_vars = sorted(
            outcome_vars, key=lambda x: env.topological_order.index(x)
        )
        col_tuples = (
            [("state", var) for var in state_vars if var not in outcome_vars]
            + [("outcome", var) for var in outcome_vars]
            + [("actual_causes", "")]
        )
        columns = pd.MultiIndex.from_tuples(col_tuples)
        ac_table = pd.DataFrame(columns=columns)

        # Get supports and variable types for all variables that depend on exogenous noise
        noise_supports = {}
        for var_name, var in env.variables.items():
            if var_name in env.structural_functions:
                if env.structural_functions[var_name].noise_dist is not None:
                    if var.var_type == "bool":
                        noise_supports[var_name] = [0, 1]
                    elif var.var_type == "int":
                        noise_supports[var_name] = list(
                            range(var.support[0], var.support[1] + 1)
                        )
                    elif var.var_type == "discrete":
                        noise_supports[var_name] = var.support
                    else:
                        raise ValueError(
                            f"get_all_actual_causes is only supported for discrete SCMs, {var_name} is {var.var_type}."
                        )

        # Get all possible configurations of noise variable values using itertools product
        # Each configuration is a dict with variable names as keys and noise values as values
        noise_configs = []
        noise_vars = list(noise_supports.keys())
        for noise_vals in itertools.product(*noise_supports.values()):
            noise_configs.append(dict(zip(noise_vars, noise_vals)))

        # For each value of noise variables, generate the state and find actual causes of the outcome
        for noise in noise_configs:

            # Get the state for the given noise configuration
            state = env.get_state(noise)
            # Get the outcome for the given state
            outcome = {}
            for var in outcome_vars:
                outcome[var] = state[var]
            # Find all actual causes for the outcome in the given state
            actual_causes = self.solve(state, outcome, noise)

            # Add the state, outcome and actual causes to the table
            # One column each for state, outcome, and actual causes
            # Under state, one column for each state variable not in the outcome
            # Under outcome, one column for each outcome variable
            # Under actual causes, just the list of actual causes
            # Each row is a different state and list of actual causes
            # Always follow topological order

            combined_row = {
                ("state", k): v for k, v in state.items() if k not in outcome_vars
            }
            combined_row.update({("outcome", k): v for k, v in outcome.items()})
            combined_row[("actual_causes", "")] = list(actual_causes.keys())
            # Add row to table
            ac_table = ac_table._append(combined_row, ignore_index=True)

        return ac_table

    def get_actual_cause(
        self,
        state: dict,
        outcome: dict,
        noise: dict = None,
    ):
        """
        Find one actual cause of the outcome in the state
        :param state: dictionary of values of all observable variables
        :param outcome: dictionary of values of the outcome variables
        :param noise: dictionary of values of all exogenous noise variables
        :return: actual_causes: list of variable names or tuples of variable names whose values are actual causes
        :return: info: dict with additional info about the actual causality test
        """
        pass

    def solve(
        self,
        state: dict,
        outcome: dict,
        noise: dict = None,
    ):
        """
        Find all actual causes of the outcome in a given state
        :param env: StructuralCausalModel
        :param outcome: dictionary of values of the outcome variables
        :param noise: dictionary of values of all exogenous noise variables
        :return: actual_causes: list of variable names or tuples of variable names whose values are actual causes
        :return: info: dict with additional info about the actual causality test
        """
        raise NotImplementedError
