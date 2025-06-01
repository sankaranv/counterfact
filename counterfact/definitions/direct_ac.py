from counterfact.causal_models.scm import StructuralCausalModel
from counterfact.definitions import ACDefinition
import numpy as np
from counterfact.inference import *
from counterfact.utils.subsets import get_all_subsets


class DirectActualCause(ACDefinition):

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
        Check if the event satisfies contrastive necessary under weak sufficiency given a witness set
        We need to find an alternative event that is not weakly sufficient for the outcome
        :param env:
        :param event:
        :param outcome:
        :param state:
        :param noise:
        :param witness_set:
        :return:
        """
        info = {"necessity_defn": "ContrastiveNecessity"}

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
        Check if the event is directly sufficient for the outcome in the state
        :param env:
        :param event:
        :param outcome:
        :param state:
        :param noise:
        :return:
        """

        # Build the witness
        if "witness_set" in kwargs:
            witness = {var: state[var] for var in kwargs["witness_set"]}
        else:
            witness = None

        # Optional: filter out non-parents since only parents can be directly sufficient
        # For proof, see Proposition 5 in Causal Sufficiency and Actual Causation, Beckers 2021
        # if "filter_non_parents" in kwargs and kwargs["filter_non_parents"]:
        #     # Get parents of all outcome variables
        #     parents = set()
        #     for var in outcome:
        #         parents.add(env.causal_graph.predecessors(var))
        #     for event_var in event:
        #         if event_var not in parents:
        #             return False, {}

        info = {"sufficiency_defn": "DirectSufficiency"}
        all_vars = list(env.variables.keys())
        event_vars = list(event.keys())
        outcome_vars = list(outcome.keys())
        remaining_vars = list(set(all_vars) - set(event_vars) - set(outcome_vars))

        # Collect supports for all variables not in the witness or event
        supports = []
        for var_name in state:

            # Ignore the variables in the event or witness set
            if var_name in event:
                continue
            if witness is not None and var_name in witness:
                continue

            # Collect the support for the variable
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
        rem_var_combinations = np.array(np.meshgrid(*supports)).T.reshape(
            -1, len(supports)
        )
        np.random.shuffle(rem_var_combinations)

        # Check if the observed outcome is produced for all combinations along with the event and witness
        for rem_var_assignment in rem_var_combinations:

            # Set up the intervention
            rem_var_intervention = {
                var: value for var, value in zip(remaining_vars, rem_var_assignment)
            }
            env.intervene(rem_var_intervention)

            # Intervene on the model to apply the given event and witness
            env.intervene(event)
            if witness is not None:
                env.intervene(witness)

            # Check if the outcome is satisfied
            new_state = env.get_state(noise)
            for var in outcome:
                if new_state[var] != outcome[var]:
                    info["ac2b_alt_state"] = new_state
                    info["ac2b_alt_outcome"] = {v: new_state[v] for v in outcome}
                    env.reset()
                    return False, info

        # All possible interventions on the remaining variables were sufficient for the outcome
        # Reset the model to its original state and return result
        env.reset()
        return True, info
