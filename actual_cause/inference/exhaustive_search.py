import numpy as np
from actual_cause.inference.solver import ACSolver


class ExhaustiveSearch(ACSolver):

    def __init__(self, env, ac_defn):

        super().__init__(env, ac_defn)

        # Check if all variables are binary or discrete or int with finite support
        # Otherwise raise an error
        for var in env.variables:
            if env.variables[var].type not in ["bool", "discrete", "int"]:
                raise ValueError(
                    f"Variable {var} is not binary or discrete or int, cannot use exhaustive search"
                )
            if env.variables[var].type == "int":
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
        all_vars = list(set(self.env.variables.keys()) - set(outcome_vars))
        actual_causes = []
        attempted_events = {
            size: set() for size in range(1, len(self.env.variables) + 1)
        }
        # Try all possible events and check if they are actual causes under some witness
        # Try random subsets of increasing size as candidate causes
        for size in range(1, len(self.env.variables) + 1):

            # Check if we have already tried all events of the given size
            # Calculate the number of possible events of the given size
            num_possible_events = np.prod([len(all_vars) - i for i in range(size)])
            if len(attempted_events[size]) == num_possible_events:
                break

            # Pick a random subset of variables of the given size
            event_vars = np.random.choice(all_vars, size, replace=False)
            remaining_vars = list(set(all_vars) - set(event_vars))
            event = {var: state[var] for var in event_vars}

            # Check if the event has already been attempted
            event_tuple = tuple(event.items())
            if event_tuple in attempted_events:
                continue

            # Add event to set of attempted events
            attempted_events[size].add(event_tuple)

            # Check if the event is an actual cause without a witness set
            if self.env.is_actual_cause(event, outcome, state, noise):
                actual_causes.append((event, []))
            else:

                # Try all possible witness sets and see if the event is an actual cause with a witness set
                witness = []
                attempted_witness_sets = {
                    size: set() for size in range(1, len(remaining_vars) + 1)
                }
                found_witness = False
                while not found_witness:

                    # Stop if we have already tried all possible witness sets
                    if len(attempted_witness_sets) == 2 ** len(remaining_vars):
                        break

                    # Try subsets of increasing size
                    witness_size = 1
                    while witness_size < len(remaining_vars):

                        # Check if all possible witnesses of the given size have been found
                        num_possible_witnesses = np.prod(
                            [len(remaining_vars) - i for i in range(witness_size)]
                        )
                        if (
                            len(attempted_witness_sets[witness_size])
                            == num_possible_witnesses
                        ):
                            witness_size += 1
                            continue

                        # Pick a random subset of variables of the given size
                        witness_vars = np.random.choice(
                            remaining_vars, witness_size, replace=False
                        )

                        # Check if the witness set has already been attempted
                        witness_tuple = tuple(witness_vars)
                        if witness_tuple in attempted_witness_sets[witness_size]:
                            continue

                        # Add witness set to set of attempted witness sets
                        attempted_witness_sets[witness_size].add(witness_tuple)

                        # Intervene on the witness set variables to set them to their actual values
                        witness = {var: state[var] for var in witness_vars}
                        self.env.intervene(witness)

                        # Check if the event is an actual cause under the given witness
                        if self.env.is_actual_cause(event, outcome, state, noise):
                            actual_causes.append((event, witness))

                        # Reset the effect of the intervention on the witness set variables
                        self.env.reset()

    def find_necessary_event(self, event, state, outcome, noise=None):

        # Collect lists for event, outcome, and remaining variables
        event_vars = list(event.keys())
        outcome_vars = list(outcome.keys())
        remaining_vars = list(
            set(self.env.variables.keys()) - set(event_vars) - set(outcome_vars)
        )

        # Obtain supports for every event variable as sets
        supports = {}
        for var_name in event_vars:
            var = self.env.variables[var_name]
            if var.var_type == "bool":
                supports[var_name] = [0, 1]
            elif var.var_type == "int":
                supports[var_name] = list(range(var.support[0], var.support[1] + 1))
            elif var.var_type == "discrete":
                supports[var_name] = list(var.support)

        # Loop until all possible witness sets and alt events have been tried
        attempted_alt_events = set()
        found_necessary_event = False
        while not found_necessary_event:

            # Try to generate an alt event
            generated_alt_event = False
            generated_witness_set = False
            alt_event = {}
            witness_set = []
            while not generated_alt_event:

                # Stop if we have already tried all possible alt events
                if len(attempted_alt_events) == 2 ** len(event_vars):
                    break

                for var_name in event_vars:
                    # On first attempt, flip values of all binary event variables
                    if len(attempted_alt_events) == 0 and len(supports[var_name]) == 2:
                        # Pick the other value in the support
                        alt_event[var_name] = supports[var_name][
                            np.abs(event[var_name] - 1)
                        ]
                    else:
                        alt_event[var_name] = np.random.choice(supports[var_name])

                # Check if the alt event has already been attempted
                alt_event_tuple = tuple(alt_event.items())
                if alt_event_tuple in attempted_alt_events:
                    continue

                # Check if the alt event is equal to the observed event
                if alt_event == event:
                    continue

                # Add alt event to set of attempted alt events
                generated_alt_event = True
                attempted_alt_events.add(alt_event_tuple)

                # Try random subsets of increasing size as witness sets
                attempted_witness_sets = set()
                generated_witness_set = False

                # For a newly generated event, reset the flag for generated witness set and try to find a new one
                while not generated_witness_set:

                    # Stop if we have already tried all possible witness sets
                    if len(attempted_witness_sets) == 2 ** len(remaining_vars):
                        break

                    # On first attempt, leave the witness set empty
                    if len(attempted_witness_sets) == 0:
                        generated_witness_set = True
                        break

                    # Try subsets of increasing size
                    for size in range(1, len(remaining_vars) + 1):

                        # Pick a random subset of variables of the given size
                        witness_set = np.random.choice(
                            remaining_vars, size, replace=False
                        )

                        # Return the witness set if it was not already attempted
                        witness_set_tuple = tuple(witness_set)
                        if witness_set_tuple not in attempted_witness_sets:
                            generated_witness_set = True
                            attempted_witness_sets.add(witness_set_tuple)
                            break

                # If a witness set could not be found with this alt event, skip it
                if not generated_witness_set:
                    continue

            # If an alt event was not found, stop
            if not generated_alt_event:
                break

            # Add the actual values of the witness set variables to the alt event
            for var_name in witness_set:
                alt_event[var_name] = state[var_name]

            # Intervene on the env to set the values in the alt event and witness set
            self.env.intervene(alt_event)

            # Check if the event is necessary
            alt_state = self.env.get_state(noise)
            alt_outcome = {var: alt_state[var] for var in outcome_vars}
