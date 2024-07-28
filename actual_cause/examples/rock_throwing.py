from actual_cause.causal_models.scm import StructuralCausalModel, StructuralFunction
from actual_cause.causal_models.variables import Variable, ExogenousNoise
import numpy as np


class RockThrowing(StructuralCausalModel):

    def __init__(self):
        super().__init__()

        # Add all variables

        self.add_variables(
            [
                Variable("suzy_throws", "bool"),
                Variable("billy_throws", "bool"),
                Variable("suzy_hits", "bool"),
                Variable("billy_hits", "bool"),
                Variable("bottle_shatters", "bool"),
            ]
        )

        # Create structural functions
        def suzy_throws(inputs, noise):
            return noise

        def billy_throws(inputs, noise):
            return noise

        def suzy_hits(inputs, noise):
            return inputs["suzy_throws"]

        def billy_hits(inputs, noise):
            return inputs["billy_throws"] * (1 - inputs["suzy_hits"])

        def bottle_shatters(inputs, noise):
            return int(np.logical_or(inputs["suzy_hits"], inputs["billy_hits"]))

        # Set structural functions for the model

        self.set_structural_functions(
            {
                "suzy_throws": StructuralFunction(
                    suzy_throws,
                    [],
                    ExogenousNoise(
                        "u_suzy_throws", lambda: np.random.choice([0, 1], p=[0.5, 0.5])
                    ),
                ),
                "billy_throws": StructuralFunction(
                    billy_throws,
                    [],
                    ExogenousNoise(
                        "u_billy_throws", lambda: np.random.choice([0, 1], p=[0.5, 0.5])
                    ),
                ),
                "suzy_hits": StructuralFunction(suzy_hits, ["suzy_throws"], None),
                "billy_hits": StructuralFunction(
                    billy_hits, ["billy_throws", "suzy_hits"], None
                ),
                "bottle_shatters": StructuralFunction(
                    bottle_shatters, ["suzy_hits", "billy_hits"], None
                ),
            }
        )
