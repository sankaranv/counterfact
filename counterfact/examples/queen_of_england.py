from counterfact.causal_models.scm import StructuralCausalModel, StructuralFunction
from counterfact.causal_models.variables import Variable, ExogenousNoise
import numpy as np


class QueenOfEngland(StructuralCausalModel):
    def __init__(self):
        super().__init__()
        self.add_variables(
            [
                Variable("queen", "discrete", support=[-1, 0, 1]),
                Variable("gardener", "bool"),
                Variable("flowers_live", "bool"),
            ]
        )

        def queen(inputs, noise):
            return noise

        def gardener(inputs, noise):
            return noise

        def flowers_live(inputs, noise):
            return inputs["gardener"] or inputs["queen"] == 1

        self.set_structural_functions(
            {
                "queen": StructuralFunction(
                    queen,
                    [],
                    ExogenousNoise("u_queen", lambda: np.random.choice([-1, 0, 1])),
                ),
                "gardener": StructuralFunction(
                    gardener,
                    [],
                    ExogenousNoise(
                        "u_gardener", lambda: np.random.choice([0, 1], p=[0.5, 0.5])
                    ),
                ),
                "flowers_live": StructuralFunction(
                    flowers_live, ["queen", "gardener"], None
                ),
            }
        )
