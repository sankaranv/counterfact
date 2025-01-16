import numpy as np
from actual_cause.causal_models.scm import StructuralCausalModel
from actual_cause.definitions.ac_definition import ACDefinition
from actual_cause.inference.solver import ACSolver
from actual_cause.utils import get_all_subsets


class HPExhaustiveSearch(ACSolver):

    def __init__(self, env, ac_defn):

        from actual_cause.definitions import ModifiedHP

        super().__init__(env, ac_defn)

        # Check if all variables are binary or discrete or int with finite support
        for var in env.variables:
            if env.variables[var].var_type not in ["bool", "discrete", "int"]:
                raise ValueError(
                    f"Variable {var} is not binary or discrete or int, cannot use exhaustive search"
                )
            if env.variables[var].var_type == "int":
                if not np.isfinite(env.variables[var].support[0]) or not np.isfinite(
                    env.variables[var].support[1]
                ):
                    raise ValueError(
                        f"Variable {var} is not int with finite support, cannot use exhaustive search"
                    )

    def solve(self, state, outcome, noise=None):

        # Collect lists for event, outcome, and remaining variables
        outcome_vars = list(outcome.keys())
        state_vars = list(state.keys())
        remaining_vars = list(set(state_vars) - set(outcome_vars))
        actual_causes = {}

        # Get all possible subsets of variables whose values can be candidate causes
        all_subsets = get_all_subsets(remaining_vars, shuffle_by_size=True)

        for subset in all_subsets:
            # Check if the event is a superset of a prior actual cause
            # If not, it will fail AC3 anyway and cannot be an actual cause
            if any([set(subset).issubset(set(ac)) for ac in actual_causes]):
                continue

            # Check if the event is an actual cause
            event = {var: state[var] for var in subset}
            is_actual_cause, info = self.ac_defn.is_actual_cause(
                self.env, event, outcome, state, noise
            )
            if is_actual_cause:
                actual_causes[subset] = {"event": event, "info": info}

        return actual_causes


# class IVPExhaustiveSearch(ACSolver):

#     def __init__(self, env: StructuralCausalModel, ac_defn: ACDefinition):

#         if not isinstance(ac_defn, FunctionalActualCause):
#             raise ValueError(
#                 "IVPExhaustiveSearch is only supported for the FunctionalActualCause definition"
#             )
#         super().__init__(env, ac_defn)

#         # Check if all variables are binary or discrete or int with finite support
#         for var in env.variables:
#             if env.variables[var].var_type not in ["bool", "discrete", "int"]:
#                 raise ValueError(
#                     f"Variable {var} is not binary or discrete or int, cannot use exhaustive search"
#                 )
#             if env.variables[var].var_type == "int":
#                 if not np.isfinite(env.variables[var].support[0]) or not np.isfinite(
#                     env.variables[var].support[1]
#                 ):
#                     raise ValueError(
#                         f"Variable {var} is not int with finite support, cannot use exhaustive search"
#                     )

#     def solve(self):
#         """
#         Find all possible partitions of the state space into IVPs. Each IVP in the partition should satisfy the following conditions
#         Contextual Invariance: For any given value of the IVP variables, all other states where the IVP variables have the same values should have the same outcome
#         Necessity: For every state in the IVP, the IVP variables' assignment should be necessary for the outcome
#         Minimality: The cost of the partition should be minimized
#         """

#         # Modified HP
#         modified_hp = ModifiedHP()

#         # Get all variables in the environment
#         all_vars = list(self.env.variables.keys())

#         # Collect supports for all variables
#         supports = []
#         for var_name in all_vars:
#             var_support = self.env.variables[var_name].support
#             var_type = self.env.variables[var_name].var_type
#             if var_type == "float":
#                 raise ValueError(
#                     f"Modified HP is not supported for float variable {var_name}"
#                 )
#             elif var_type == "int":
#                 supports.append(np.arange(var_support[0], var_support[1] + 1, 1))
#             else:
#                 supports.append(var_support)

#         # Enumerate all possible states that the environment can generate
#         all_states = np.array(np.meshgrid(*supports)).T.reshape(-1, len(supports))

#         # For every state, find all events that are necessary for the outcome
#         necessary_events_per_state = {}
#         for state in all_states:
#             state_dict = {all_vars[i]: state[i] for i in range(len(all_vars))}
#             outcome = self.env.get_outcome(state_dict)
#             necessary_events_per_state[tuple(state)] = set()

#             # For every subset of variables, check if it is necessary for the outcome in the state
#             for event_vars in get_all_subsets(all_vars, include_full=True):
#                 event = {var: state_dict[var] for var in event_vars}
#                 necessary, info = modified_hp.is_necessary(
#                     self.env, event, outcome, state_dict
#                 )
#                 if necessary:
#                     necessary_events_per_state[tuple(state)].add(tuple(event_vars))

#         # For every subset of states, collect all variable subsets that satisfy necessity
#         candidate_ivp_assignments = {}
#         for state_subset in get_all_subsets(all_states, include_full=True):
#             # Intersection of all necessary events for the states in the subset
#             candidate_ivp_assignments[state_subset] = set.intersection(
#                 *[necessary_events_per_state[state] for state in state_subset]
#             )

#         # Find all partitions of the state space into n partitions where n is the number of IVPs
#         num_ivps = 2 ** len(all_vars)
#         ivp_partitions = []
