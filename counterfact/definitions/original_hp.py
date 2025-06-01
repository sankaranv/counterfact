from counterfact.definitions import ACDefinition
from counterfact.causal_models.scm import StructuralCausalModel
import numpy as np
from counterfact.inference import *
from counterfact.utils.subsets import get_all_subsets


class OriginalHP(ACDefinition):

    def __init__(self):
        super().__init__()

    def is_necessary(
        self,
        env: StructuralCausalModel,
        event: dict,
        outcome: dict,
        state: dict,
        noise=None,
        **kwargs,
    ):
        """
        There is a partition of the set of state variables S into two sets Z and W, where X is contained in Z
        For this partition, there is an intervention X=x' and W=w* under which the outcome is not satisfied
        :param env:
        :param event:
        :param outcome:
        :param state:
        :param noise:
        :param witness_set:
        :return:
        """
        info = {"necessity_defn": "OriginalHP_AC2a"}

        # Build the witness
        if "witness_set" in kwargs:
            witness = {var: state[var] for var in kwargs["witness_set"]}
        else:
            witness = None

        all_vars = list(env.variables.keys())
        event_vars = list(event.keys())
        outcome_vars = list(outcome.keys())
        remaining_vars = list(set(all_vars) - set(event_vars) - set(outcome_vars))

        # Collect supports for all variables in the event
        supports = []
        for var_name in event:
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

        # Shuffle the event combinations
        np.random.shuffle(event_combinations)

        # Check if any of the combinations are not sufficient for the outcome
        for alt_assignment in event_combinations:

            # Ignore the original event
            if np.array_equal(alt_assignment, original_assignment):
                continue

            # Set up alternative event
            alt_event = {var: value for var, value in zip(event_vars, alt_assignment)}

            if witness is not None:

                # Reset the effect of prior interventions
                env.reset()

                # Apply the witness set intervention if a witness set is provided
                env.intervene(witness)

                # Apply the alternative event as an intervention on the causal model to obtain an alternate outcome
                env.intervene(alt_event)
                alt_state = env.get_state(noise)
                alt_outcome = {var: alt_state[var] for var in outcome}
                # Check if the sufficiency condition is violated by the alternative event and outcome
                sufficient, ac2b_info = self.is_sufficient(
                    env, alt_event, outcome, alt_state, noise
                )
                if not sufficient:
                    info["ac2a_alt_event"] = alt_event
                    info["ac2a_witness"] = witness
                    env.reset()
                    return True, info

            else:
                # No witness provided, so we try all possible witness sets
                for witness_set in get_all_subsets(
                    remaining_vars, shuffle_by_size=True
                ):

                    # Reset the effect of prior interventions
                    env.reset()

                    # Ignore the witness set that is the same as the original event
                    witness = {var: state[var] for var in witness_set}
                    if set(witness_set) == set(event_vars):
                        continue

                    # Apply the witness set intervention
                    env.intervene(witness)

                    # Apply the intervention on the causal model to obtain an alternate outcome
                    alt_event = {
                        var: value for var, value in zip(event_vars, alt_assignment)
                    }
                    env.intervene(alt_event)

                    # Print the functions of the SCM that are constant functions
                    alt_state = env.get_state(noise)
                    alt_outcome = {var: alt_state[var] for var in outcome}

                    # Check if the sufficiency condition is violated by the alternative event and outcome
                    sufficient, ac2b_info = self.is_sufficient(
                        env, alt_event, outcome, alt_state, noise
                    )
                    if not sufficient:

                        # Collect information
                        info["ac2a_alt_event"] = alt_event
                        info["ac2a_witness"] = witness
                        if "ac2b_alt_outcome" in ac2b_info:
                            info["ac2a_alt_outcome"] = ac2b_info["ac2b_alt_outcome"]
                        # Reset the model to its original state and return result
                        env.reset()
                        return True, info

                # Reset witness when moving to the next alt event
                witness = None

        # No other intervention on the event variables was insufficient for the observed outcome
        # Reset the model to its original state and return result
        return False, info

    def is_sufficient(self, env, event, outcome, state, noise=None, **kwargs):
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

        info = {"sufficiency_defn": "WeakSufficiency"}
        # Build the witness
        if "witness_set" in kwargs:
            witness = {var: state[var] for var in kwargs["witness_set"]}
        else:
            witness = None

        # Intervene on the model to apply the given event
        env.intervene(event)

        # Intervene on the model to apply the witness
        if witness is not None:
            env.intervene(witness)

        # Check if the outcome is satisfied
        new_state = env.get_state(noise)
        for var in outcome:
            if new_state[var] != outcome[var]:
                info["ac2b_alt_outcome"] = {v: new_state[v] for v in outcome}
                env.reset()
                return False, info

        env.reset()
        return True, info
