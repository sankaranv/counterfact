from actual_cause.definition import ActualCauseDefinition
import pymc as pm


class ModifiedHP(ActualCauseDefinition):
    def __init__(self):
        super().__init__()

    def is_sufficient(self, env, obs, event, outcome, witness=None):
        """
        Check if the candidate is weakly sufficient for the outcome
        :param env: causal model or environment
        :param obs: observed state vector
        :param event: dict of variable values that are a candidate actual cause
        :param outcome: dict of variable values that are the outcome
        :param witness: dict of variable values that are a witness
        :return: is_necessary: bool
        """

        # We intervene on the witness and event variables and check if they correspond to the observed state

        # Combine the event and outcome dictionaries
        intervention = {**event, **outcome}

        # Apply the intervention to the environment
        intervened_model = env.do(intervention)

        # Sample from the intervened environment
        # TODO - this does not hold up for probabilistic settings!
        with intervened_model:
            trace_intervened = pm.sample(draws=1)
        intervened_obs = {
            key: trace_intervened.posterior.data_vars[key].data[0, 0]
            for key in trace_intervened.posterior.data_vars.keys()
        }
        print(f"Intervened observation: {intervened_obs}")

        # Check if the outcome is the same under intervention
        for var in outcome.keys():
            if intervened_obs[var] != outcome[var]:
                return False
        return True
