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
        original_event = np.array([event[var] for var in event_vars])
        event_combinations = np.array(np.meshgrid(*supports)).T.reshape(
            -1, len(supports)
        )

        # Check if any of the combinations are not sufficient for the outcome
        for alt_event in event_combinations:

            # Reset the effect of prior interventions
            env.reset()

            # Ignore the original event
            if np.array_equal(alt_event, original_event):
                continue

            # Apply the intervention on the causal model to obtain an alternate outcome
            env.intervene({var: value for var, value in zip(event_vars, alt_event)})
            alt_state = env.get_state(noise)

            # Check if the intervention produced any deviation from the original outcome
            for var in outcome:
                if alt_state[var] != outcome[var]:
                    return True

        # The outcome did not change under any other intervention on the event variables
        return False

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

        # TODO - needs to be replaced with interventional version, sometimes invalid states could be input

        if self.is_factual(env, event, outcome, state, noise=None):
            return True
        return False
