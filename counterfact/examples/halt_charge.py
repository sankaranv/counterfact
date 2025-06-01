from counterfact.causal_models.scm import StructuralCausalModel, StructuralFunction
from counterfact.causal_models.variables import Variable, ExogenousNoise
import numpy as np


class HaltOrCharge(StructuralCausalModel):
    def __init__(self):
        super().__init__()
        self.add_variables(
            [
                Variable("major", "int"),
                Variable("sergeant", "bool"),
                Variable("corporal", "int"),
            ]
        )

        def major(inputs, noise):
            return noise

        def sergeant(inputs, noise):
            return noise

        def corporal(inputs, noise):
            return inputs["sergeant"] if inputs["major"] == 2 else inputs["major"]

        self.set_structural_functions(
            {
                "major": StructuralFunction(
                    major,
                    [],
                    ExogenousNoise("u_major", lambda: np.random.choice([0, 1, 2])),
                ),
                "sergeant": StructuralFunction(
                    sergeant,
                    [],
                    ExogenousNoise(
                        "u_sergeant", lambda: np.random.choice([0, 1], p=[0.5, 0.5])
                    ),
                ),
                "corporal": StructuralFunction(corporal, ["major", "sergeant"], None),
            }
        )
