import numpy as np
from actual_cause.inference.solver import ACSolver
from actual_cause.utils import get_all_subsets


class HPExhaustiveSearch(ACSolver):

    def __init__(self, env, ac_defn):

        super().__init__(env, ac_defn)

        # Check if all variables are binary or discrete or int with finite support
        # Otherwise raise an error
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

        self.env = env
        self.ac_defn = ac_defn

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
