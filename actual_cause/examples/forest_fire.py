from actual_cause.causal_models.scm import StructuralCausalModel, StructuralFunction
import torch
import pyro.distributions as dist


class ForestFireDisjunctive(StructuralCausalModel):
    def __init__(self):
        super().__init__()

        for var in ["lightning", "arson", "fire"]:
            self.add_variable(var, "bool", [0, 1])

        def lightning(inputs, noise):
            return noise

        def arson(inputs, noise):
            return noise

        def fire(inputs, noise):
            dtype = inputs["lightning"].dtype
            return torch.logical_or(inputs["lightning"], inputs["arson"]).to(dtype)

        self.set_structural_functions(
            {
                "lightning": StructuralFunction(lightning, [], dist.Bernoulli(0.5)),
                "arson": StructuralFunction(arson, [], dist.Bernoulli(0.5)),
                "fire": StructuralFunction(fire, ["lightning", "arson"], None),
            }
        )


class ForestFireConjunctive(StructuralCausalModel):
    def __init__(self):
        super().__init__()

        for var in ["lightning", "arson", "fire"]:
            self.add_variable(var, "bool", [0, 1])

        def lightning(inputs, noise):
            return noise

        def arson(inputs, noise):
            return noise

        def fire(inputs, noise):
            dtype = inputs["lightning"].dtype
            return torch.logical_and(inputs["lightning"], inputs["arson"]).to(dtype)

        self.set_structural_functions(
            {
                "lightning": StructuralFunction(lightning, [], dist.Bernoulli(0.5)),
                "arson": StructuralFunction(arson, [], dist.Bernoulli(0.5)),
                "fire": StructuralFunction(fire, ["lightning", "arson"], None),
            }
        )


class ForestFireRainStorm(StructuralCausalModel):
    def __init__(self):
        super().__init__()

        for var in [
            "april_showers",
            "may_electric_storm",
            "june_electric_storm",
            "fire_in_may",
            "fire_in_june",
        ]:
            self.add_variable(var, "bool", [0, 1])

        def april_showers(inputs, noise):
            return noise

        def may_electric_storm(inputs, noise):
            return noise

        def june_electric_storm(inputs, noise):
            return noise

        def fire_in_may(inputs, noise):
            dtype = inputs["may_electric_storm"].dtype
            return torch.logical_and(
                inputs["may_electric_storm"], torch.logical_not(inputs["april_showers"])
            ).to(dtype)

        def fire_in_june(inputs, noise):
            dtype = inputs["june_electric_storm"].dtype
            return torch.logical_and(
                inputs["june_electric_storm"],
                torch.logical_or(
                    inputs["april_showers"],
                    torch.logical_not(inputs["may_electric_storm"]),
                ),
            ).to(dtype)

        self.set_structural_functions(
            {
                "april_showers": StructuralFunction(
                    april_showers, [], dist.Bernoulli(0.5)
                ),
                "may_electric_storm": StructuralFunction(
                    may_electric_storm, [], dist.Bernoulli(0.5)
                ),
                "june_electric_storm": StructuralFunction(
                    june_electric_storm, [], dist.Bernoulli(0.5)
                ),
                "fire_in_may": StructuralFunction(
                    fire_in_may, ["may_electric_storm", "april_showers"], None
                ),
                "fire_in_june": StructuralFunction(
                    fire_in_june,
                    ["june_electric_storm", "april_showers", "may_electric_storm"],
                    None,
                ),
            }
        )
