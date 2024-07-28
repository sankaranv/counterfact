import pymc as pm
from pymc.math import switch, eq
import numpy as np
from actual_cause.environment import Environment


class HaltOrCharge(Environment):

    def __init__(self, n_samples_per_obs=1):
        super().__init__(n_samples_per_obs=n_samples_per_obs)

    def build_model(self):

        self.description = """In this example, orders from higher-ranking officers could trump those of lower-ranking
         officers. The major can order the corporal to halt or charge (these are indicated by values 0 and 1), or 
         choose not to make an order (indicated by a value of 2). If the Major makes no order, the corporal will defer 
         to the order of the sergeant, where 1 indicates an order to charge and 0 indicates an order to halt. The 
         corporal will always follow the order they are given"""

        with self.model:
            # Exogenous variables as priors
            u_major = pm.Dirichlet("u_major", np.array([1, 1, 1]))
            u_sergeant = pm.Beta("u_sergeant", 1, 1)

            major = pm.Categorical("major", u_major)
            sergeant = pm.Bernoulli("sergeant", u_sergeant)

            # Major = 2 indicates that they defer to the sergeant
            corporal = pm.Deterministic(
                "corporal", switch(eq(major, 2), sergeant, major)
            )
