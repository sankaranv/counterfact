import pymc as pm
from pymc.math import switch, eq, and_, or_
from actual_cause.environment import Environment


class RockThrowing(Environment):

    def __init__(self, mode="deterministic", n_samples_per_obs=1):
        super().__init__(mode=mode, n_samples_per_obs=n_samples_per_obs)
        self.description = """Suzy and Billy are throwing rocks at a bottle. The bottle shatters if either Suzy or 
        Billy hits it. Billy's throw is always slower than Suzy's, and it will hit the bottle only if Suzy's throw does
        not. This is a classic example of preemption.
        """

    def build_model(self):
        if self.mode == "deterministic":
            self.deterministic_model()
        elif self.mode == "probabilistic":
            self.probabilistic_model()
        else:
            raise ValueError(
                "Invalid mode, must be either 'deterministic' or 'probabilistic'"
            )

    def deterministic_model(self):
        with self.model:
            u_suzy_throws = pm.Beta("u_suzy_throws", 1, 1)
            u_billy_throws = pm.Beta("u_billy_throws", 1, 1)
            suzy_throws = pm.Bernoulli("suzy_throws", u_suzy_throws)
            billy_throws = pm.Bernoulli("billy_throws", u_billy_throws)
            suzy_hits = pm.Deterministic("suzy_hits", suzy_throws)
            billy_hits = pm.Deterministic("billy_hits", billy_throws * (1 - suzy_hits))
            bottle_shatters = pm.Deterministic(
                "bottle_shatters", or_(suzy_hits, billy_hits)
            )

    def probabilistic_model(self):

        with self.model:
            # Priors for all variables in the model
            u_suzy_throws = pm.Beta("u_suzy_throws", 1, 1)
            u_billy_throws = pm.Beta("u_billy_throws", 1, 1)
            u_suzy_hits = pm.Beta("u_suzy_hits", 1, 1)
            u_billy_hits = pm.Beta("u_billy_hits", 1, 1)
            u_bottle_shatters_if_suzy_hits = pm.Beta(
                "u_bottle_shatters_if_suzy_hits", 1, 1
            )
            u_bottle_shatters_if_billy_hits = pm.Beta(
                "u_bottle_shatters_if_billy_hits", 1, 1
            )

            # Use the priors to define the conditional probabilities of each one throwing
            suzy_throws = pm.Bernoulli("suzy_throws", u_suzy_throws)
            billy_throws = pm.Bernoulli("billy_throws", u_billy_throws)

            # Make sure Suzy hits with zero probability if she doesn't throw
            prob_suzy_hits = pm.math.switch(eq(suzy_throws, 1), u_suzy_hits, 0.0)
            suzy_hits = pm.Bernoulli("suzy_hits", prob_suzy_hits)

            # Make sure Billy hits with zero probability if he doesn't throw or Suzy throws
            prob_billy_hits = pm.math.switch(
                and_(eq(billy_throws, 1), eq(suzy_hits, 0)), u_billy_hits, 0.0
            )
            billy_hits = pm.Bernoulli("billy_hits", prob_billy_hits)

            # Only one rock will ever hit the bottle so we can just add the probabilities
            prob_bottle_shatters = switch(
                eq(suzy_hits, 1),
                u_bottle_shatters_if_suzy_hits,
                switch(eq(billy_hits, 1), u_bottle_shatters_if_billy_hits, 0.0),
            )

            bottle_shatters = pm.Bernoulli("bottle_shatters", prob_bottle_shatters)
