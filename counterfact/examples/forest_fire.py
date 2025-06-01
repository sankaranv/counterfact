from counterfact.causal_models.scm import StructuralCausalModel, StructuralFunction
from counterfact.causal_models.variables import Variable, ExogenousNoise
import numpy as np


class ForestFireDisjunctive(StructuralCausalModel):
    def __init__(self):
        super().__init__()
        self.add_variables(
            [
                Variable("lightning", "bool"),
                Variable("arson", "bool"),
                Variable("fire", "bool"),
            ]
        )

        def lightning(inputs, noise):
            return noise

        def arson(inputs, noise):
            return noise

        def fire(inputs, noise):
            return int(np.logical_or(inputs["lightning"], inputs["arson"]))

        self.set_structural_functions(
            {
                "lightning": StructuralFunction(
                    lightning,
                    [],
                    ExogenousNoise(
                        "u_lightning", lambda: np.random.choice([0, 1], p=[0.5, 0.5])
                    ),
                ),
                "arson": StructuralFunction(
                    arson,
                    [],
                    ExogenousNoise(
                        "u_arson", lambda: np.random.choice([0, 1], p=[0.5, 0.5])
                    ),
                ),
                "fire": StructuralFunction(fire, ["lightning", "arson"], None),
            }
        )


class ForestFireConjunctive(StructuralCausalModel):
    def __init__(self):
        super().__init__()
        self.add_variables(
            [
                Variable("lightning", "bool"),
                Variable("arson", "bool"),
                Variable("fire", "bool"),
            ]
        )

        def lightning(inputs, noise):
            return noise

        def arson(inputs, noise):
            return noise

        def fire(inputs, noise):
            return inputs["lightning"] and inputs["arson"]

        self.set_structural_functions(
            {
                "lightning": StructuralFunction(
                    lightning,
                    [],
                    ExogenousNoise(
                        "u_lightning", lambda: np.random.choice([0, 1], p=[0.5, 0.5])
                    ),
                ),
                "arson": StructuralFunction(
                    arson,
                    [],
                    ExogenousNoise(
                        "u_arson", lambda: np.random.choice([0, 1], p=[0.5, 0.5])
                    ),
                ),
                "fire": StructuralFunction(fire, ["lightning", "arson"], None),
            }
        )


class ForestFireRainStorm(StructuralCausalModel):
    def __init__(self):
        super().__init__()
        self.add_variables(
            [
                Variable("april_showers", "bool"),
                Variable("may_electric_storm", "bool"),
                Variable("june_electric_storm", "bool"),
                Variable("fire_in_may", "bool"),
                Variable("fire_in_june", "bool"),
            ]
        )

        def april_showers(inputs, noise):
            return noise

        def may_electric_storm(inputs, noise):
            return noise

        def june_electric_storm(inputs, noise):
            return noise

        def fire_in_may(inputs, noise):
            return int(inputs["may_electric_storm"] and not inputs["april_showers"])

        def fire_in_june(inputs, noise):
            return inputs["june_electric_storm"] and (
                inputs["april_showers"] or not inputs["may_electric_storm"]
            )

        self.set_structural_functions(
            {
                "april_showers": StructuralFunction(
                    april_showers,
                    [],
                    ExogenousNoise(
                        "u_april_showers",
                        lambda: np.random.choice([0, 1], p=[0.5, 0.5]),
                    ),
                ),
                "may_electric_storm": StructuralFunction(
                    may_electric_storm,
                    [],
                    ExogenousNoise(
                        "u_may_electric_storm",
                        lambda: np.random.choice([0, 1], p=[0.5, 0.5]),
                    ),
                ),
                "june_electric_storm": StructuralFunction(
                    june_electric_storm,
                    [],
                    ExogenousNoise(
                        "u_june_electric_storm",
                        lambda: np.random.choice([0, 1], p=[0.5, 0.5]),
                    ),
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
