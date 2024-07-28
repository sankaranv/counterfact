import pymc as pm
from pymc.math import and_, or_, eq
from actual_cause.environment import Environment


class ForestFire(Environment):

    def __init__(self, mode="rain_storm", n_samples_per_obs=1):
        super().__init__(mode=mode, n_samples_per_obs=n_samples_per_obs)
        self.mode = mode

    def build_model(self):
        if self.mode == "disjunctive":
            self.disjunctive_model()
        elif self.mode == "conjunctive":
            self.conjunctive_model()
        elif self.mode == "rain_storm":
            self.rain_storm_model()
        else:
            raise ValueError(
                f"Unknown model: {self.mode}. Must be one of 'disjunctive', 'conjunctive', or 'rain_storm'."
            )

    def disjunctive_model(self):

        self.description = """In this disjunctive model, a forest fire is caused either if there is a lightning strike or an arsonist 
        drops a match. The prior for lightning indicates the dryness of wood in the forest, and the prior for the 
        arsonist indicates whether there is enough oxygen to start the fire."""

        with self.model:
            u_lightning = pm.Beta("u_lightning", 1, 1)
            u_arson = pm.Beta("u_arson", 1, 1)
            lightning = pm.Bernoulli("lightning", p=u_lightning)
            arson = pm.Bernoulli("arson", p=u_arson)
            fire = pm.Deterministic("fire", or_(lightning, arson))

    def conjunctive_model(self):

        self.description = """In this conjunctive model, a forest fire is caused only if there is a lightning strike and an arsonist 
        drops a match. The prior for lightning indicates the dryness of wood in the forest, and the prior for the 
        arsonist indicates whether there is enough oxygen to start the fire."""

        with self.model:
            u_lightning = pm.Beta("u_lightning", 1, 1)
            u_arson = pm.Beta("u_arson", 1, 1)
            lightning = pm.Bernoulli("lightning", p=u_lightning)
            arson = pm.Bernoulli("arson", p=u_arson)
            fire = pm.Deterministic("fire", and_(lightning, arson))

    def rain_storm_model(self):

        self.description = """
        In this model, there will be a forest fire in May if there were electrical storms in May and no heavy rain 
        in April. In June, there will be a forest fire if there were electrical storms in June and either heavy rain 
        in April or no electrical storms in May.
        """

        with self.model:
            u_april_heavy_rain = pm.Beta("u_april_heavy_rain", 1, 1)
            u_may_electric_storm = pm.Beta("u_may_electric_storm", 1, 1)
            u_june_electric_storm = pm.Beta("u_june_electric_storm", 1, 1)
            april_heavy_rain = pm.Bernoulli("april_showers", p=u_april_heavy_rain)
            may_electric_storm = pm.Bernoulli(
                "may_electric_storm", p=u_may_electric_storm
            )
            june_electric_storm = pm.Bernoulli(
                "june_electric_storm", p=u_june_electric_storm
            )
            fire_in_may = pm.Deterministic(
                "fire_in_may", and_(may_electric_storm, 1 - april_heavy_rain)
            )
            fire_in_june = pm.Deterministic(
                "fire_in_june",
                and_(
                    june_electric_storm, or_(april_heavy_rain, 1 - may_electric_storm)
                ),
            )
