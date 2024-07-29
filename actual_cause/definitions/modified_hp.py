from actual_cause.definitions.base import ACDefinition
import numpy as np


class ModifiedHP(ACDefinition):

    def __init__(self):
        super().__init__()

    def is_necessary(self, env, event, outcome, state, noise=None):
        """
        Check if the event is contrastively necessary under weak sufficiency
        We need to find an alternative event that is not weakly sufficient for the outcome
        :param env:
        :param event:
        :param outcome:
        :param state:
        :param noise:
        :return:
        """
        info = {}
        # Collect supports for all variables in the event
        supports = []
        event_vars = []
        for var_name in event:
            event_vars.append(var_name)
            var_support = env.variables[var_name].support
            var_type = env.variables[var_name].var_type
            if var_type == "float":
                raise ValueError(
                    f"Modified HP is not supported for float variable {var_name}"
                )
            elif var_type == "int":
                supports.append(np.arange(var_support[0], var_support[1] + 1, 1))
            else:
                supports.append(var_support)

        # Get all combinations of values
        original_assignment = np.array([event[var] for var in event_vars])
        event_combinations = np.array(np.meshgrid(*supports)).T.reshape(
            -1, len(supports)
        )

        # Check if any of the combinations are not sufficient for the outcome
        for alt_assignment in event_combinations:

            # Reset the effect of prior interventions
            env.reset()

            # Ignore the original event
            if np.array_equal(alt_assignment, original_assignment):
                continue

            # Apply the intervention on the causal model to obtain an alternate outcome
            alt_event = {var: value for var, value in zip(event_vars, alt_assignment)}
            env.intervene(alt_event)
            alt_state = env.get_state(noise)

            # Check if the sufficiency condition is violated by the alternative event and outcome
            sufficient, ac2b_info = self.is_sufficient(
                env, alt_event, outcome, state, noise
            )
            if not sufficient:
                info["ac2a_alt_event"] = alt_event
                return True, info

        # No other intervention on the event variables was insufficient for the observed outcome
        return False, info

    def is_sufficient(self, env, event, outcome, state, noise=None):
        """
        Check if the event is weakly sufficient for the outcome in the state
        It can be shown that satisfying AC1 implies weak sufficiency is satisfied
        :param env:
        :param event:
        :param outcome:
        :param state:
        :param noise:
        :return:
        """

        info = {}

        # Reset the effect of prior interventions
        env.reset()

        # Intervene on the model to apply the given event
        env.intervene(event)
        new_state = env.get_state(noise)

        # Check if the outcome is satisfied
        for var in outcome:
            if new_state[var] != outcome[var]:
                info["ac2b_alt_outcome"] = {v: new_state[v] for v in outcome}
                return False, info

        return True, info
