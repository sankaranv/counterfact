from actual_cause.definitions.ac_definition import ACDefinition
import numpy as np
from actual_cause.inference import *


class ModifiedHP(ACDefinition):

    def __init__(self):
        super().__init__()

    def is_necessary_with_witness_old(
        self,
        env: StructuralCausalModel,
        event: dict,
        outcome: dict,
        state: dict,
        noise=None,
        witness_set: list = None,
    ):
        """
        Check if the event is contrastively necessary under weak sufficiency given a witness set
        We need to find an alternative event that is not weakly sufficient for the outcome
        :param env:
        :param event:
        :param outcome:
        :param state:
        :param noise:
        :param witness_set:
        :return:
        """
        info = {}

        # Build the witness
        if witness_set is None:
            witness = None
        else:
            witness = {var: state[var] for var in witness_set}

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

            # Apply the witness set intervention
            if witness:
                env.intervene(witness)

            # Ignore the original event
            if np.array_equal(alt_assignment, original_assignment):
                continue

            # Apply the intervention on the causal model to obtain an alternate outcome
            alt_event = {var: value for var, value in zip(event_vars, alt_assignment)}
            env.intervene(alt_event)
            alt_state = env.get_state(noise)
            alt_outcome = {var: alt_state[var] for var in outcome}
            # Check if the sufficiency condition is violated by the alternative event and outcome
            sufficient, ac2b_info = self.is_sufficient(
                env, alt_event, outcome, alt_state, noise
            )
            if not sufficient:
                info["ac2a_alt_event"] = alt_event
                print(f"Alt event: {alt_event}")
                print(f"Alt outcome: {alt_outcome}")
                return True, info

        # No other intervention on the event variables was insufficient for the observed outcome
        return False, info

    def is_necessary(
        self,
        env: StructuralCausalModel,
        event: dict,
        outcome: dict,
        state: dict,
        noise=None,
        witness_set: list = None,
        **kwargs,
    ):
        """
        Check if the event is contrastively necessary under weak sufficiency given a witness set
        We need to find an alternative event that is not weakly sufficient for the outcome
        :param env:
        :param event:
        :param outcome:
        :param state:
        :param noise:
        :param witness_set:

        :return:
        """
        info = {}
        event_vars = list(event.keys())
        original_assignment = [event[var] for var in event_vars]

        # Build the witness
        if witness_set is None:
            witness = None
        else:
            witness = {var: state[var] for var in witness_set}

        # Assign max number of attempts to find an alt event
        if "max_attempts" in kwargs:
            max_attempts = kwargs["max_attempts"]
        else:
            max_attempts = 1000

        # We will find a random alt event whose value is insufficient for the outcome
        attempted_alt_events = set()
        attempted_alt_events.add(tuple(original_assignment))

        while len(attempted_alt_events) < max_attempts:

            # Reset the effect of prior interventions
            env.reset()

            # Generate a random alternative event
            alt_event = {}
            for var_name in event_vars:
                var = env.variables[var_name]
                if var.var_type == "int":
                    alt_event[var_name] = np.random.choice(
                        range(var.support[0], var.support[1] + 1)
                    )
                elif var.var_type in ["discrete", "bool"]:
                    alt_event[var_name] = np.random.choice(var.support)
                elif var.var_type == "float":
                    alt_value = np.random.uniform(var.support[0], var.support[1])
                    if "rounding" in kwargs:
                        alt_value = round(alt_value, kwargs["rounding"])
                    else:
                        alt_value = round(alt_value, 2)
                    alt_event[var_name] = alt_value

            # Check if the alt event has already been attempted
            alt_event_tuple = tuple(alt_event.items())
            if alt_event_tuple in attempted_alt_events:
                continue

            # If witness is available, apply the witness set intervention
            env.intervene(witness)

            # Apply the intervention on the causal model to obtain an alternate outcome
            env.intervene(alt_event)
            alt_state = env.get_state(noise)
            alt_outcome = {var: alt_state[var] for var in outcome}

            # Check if the sufficiency condition is violated by the alternative event and outcome
            sufficient, ac2b_info = self.is_sufficient(
                env, alt_event, outcome, alt_state, noise
            )
            if not sufficient:
                info["ac2a_alt_event"] = alt_event
                print(f"Alt event: {alt_event}")
                print(f"Alt outcome: {alt_outcome}")
                return True, info

        # No other intervention on the event variables was insufficient for the observed outcome
        return False, info

    def is_sufficient(
        self, env, event, outcome, state, noise=None, witness_set=None, solver=None
    ):
        """
        Check if the event is weakly sufficient for the outcome in the state
        :param env:
        :param event:
        :param outcome:
        :param state:
        :param noise:
        :param witness_set:
        :param solver: None, since no solver is required for this definition
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
