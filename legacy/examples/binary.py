import pymc as pm
from pymc.math import and_, or_, neq
from actual_cause.environment import Environment


class BinaryAnd(Environment):
    def __init__(self, n_samples_per_obs=1):
        super().__init__(n_samples_per_obs=n_samples_per_obs)

    def build_model(self):
        self.description = """
        In this model, the result is true if and only if both a and b are true. The priors for a and b are independent.
        """

        with self.model:
            u_a = pm.Beta("u_a", 1, 1)
            u_b = pm.Beta("u_b", 1, 1)
            a = pm.Bernoulli("a", u_a)
            b = pm.Bernoulli("b", u_b)
            result = pm.Deterministic("result", and_(a, b))


class BinaryOr(Environment):
    def __init__(self, n_samples_per_obs=1):
        super().__init__(n_samples_per_obs)

    def build_model(self):
        self.description = """
        In this model, the result is true if either a or b are true. The priors for a and b are independent.
        """

        with self.model:
            u_a = pm.Beta("u_a", 1, 1)
            u_b = pm.Beta("u_b", 1, 1)
            a = pm.Bernoulli("a", u_a)
            b = pm.Bernoulli("b", u_b)
            result = pm.Deterministic("result", or_(a, b))


class BinaryXor(Environment):
    def __init__(self, n_samples_per_obs=1):
        super().__init__(n_samples_per_obs)

    def build_model(self):

        self.description = """
            In this model, the result is true if and only if a and b are different. The priors for a and b are independent.
            """

        with self.model:
            u_a = pm.Beta("u_a", 1, 1)
            u_b = pm.Beta("u_b", 1, 1)
            a = pm.Bernoulli("a", u_a)
            b = pm.Bernoulli("b", u_b)
            result = pm.Deterministic("result", neq(a, b))
