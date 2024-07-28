import numpy as np


class ActualCauseDefinition:

    def __init__(self):
        pass

    def is_actual_cause(self, env, obs, event, outcome, witness=None):
        """
        Check if the event is an actual cause of the outcome
        :param env: causal model or environment
        :param obs: observed state vector
        :param event: dict of variable values that are a candidate actual cause
        :param outcome: dict of variable values that are the outcome
        :param witness: dict of variable values that are a witness
        :return: is_actual_cause: bool, info: dict of boolean values for factual, necessary, sufficient, minimal
        """

        info = {
            "factual": False,
            "necessary": False,
            "sufficient": False,
            "minimal": False,
        }
        is_actual_cause = False
        # Check AC1
        if self.is_factual(env, obs, event, outcome, witness):
            info["factual"] = True
            # Check AC2b
            if self.is_sufficient(env, obs, event, outcome, witness):
                info["sufficient"] = True
                # Check AC2a
                if self.is_necessary(env, obs, event, outcome, witness):
                    info["necessary"] = True
                    # Check AC3
                    if self.is_minimal(env, obs, event, outcome, witness):
                        info["minimal"] = True
                        is_actual_cause = True
        return is_actual_cause, info

    def is_factual(self, env, obs, event, outcome, witness=None):

        # Check if the event actually happened
        for var in event.keys():
            if event[var] != obs[var]:
                return False

        # Check if the outcome actually happened
        for var in outcome.keys():
            if outcome[var] != obs[var]:
                return False

        # Check if the witness actually happened
        if witness is not None:
            for var in witness.keys():
                if witness[var] != obs[var]:
                    return False
        return True

    def is_sufficient(self, env, obs, event, outcome, witness=None):
        raise NotImplementedError

    def is_necessary(self, env, obs, event, outcome, witness=None, num_trials=1000):
        """
        Check for contrastive necessity using the supplied definition of sufficiency
        We fix the witness and search for an alternative event that could have changed the outcome
        :param env:
        :param obs:
        :param event:
        :param outcome:
        :param witness:
        :return:
        """

        # Sample from the observational distribution of X
        # TODO - this could lead to faithfulness violations slipping through the cracks
        with env.model:
            trace = env.sample(draws=num_trials)

        # For each sample, intervene and check if a different outcome is returned
        for i in range(num_trials):

            # Get a sample from the trace for the event variables only
            intervention = {key: trace[key][i] for key in event.keys()}

            # Add witness to the intervention
            if witness is not None:
                intervention = {**intervention, **witness}

            # Apply the intervention to the environment
            intervened_env = env.do(intervention)

            # Sample from the intervened environment
            intervened_obs = intervened_env.observe()

            # Check if the intervened observation matches the actual observation
            for var in intervened_obs.keys():
                if intervened_obs[var] != outcome[var]:
                    return True
        return False

    def is_minimal(self, env, obs, event, outcome, witness=None):
        """
        Check if the event is minimal
        :param env:
        :param obs:
        :param event:
        :param outcome:
        :param witness:
        :return:
        """
        return True
