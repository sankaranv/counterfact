from counterfact.causal_models.scm import StructuralCausalModel, StructuralFunction
from counterfact.causal_models.variables import Variable, ExogenousNoise
import numpy as np

# next_mover_pos = mover + 1 if obstacle != mover + 1 else mover


class Mover1D(StructuralCausalModel):
    def __init__(self, world_length: int = 4):
        super().__init__()
        self.add_variables(
            [
                Variable("mover", "int", support=[0, world_length - 1]),
                Variable("obstacle", "int", support=[0, world_length]),
                Variable("next_mover_pos", "int", support=[1, world_length]),
            ]
        )

        self.formatted_var_names = {
            "mover": "$m$",
            "obstacle": "$o$",
            "next_mover_pos": "$m^\prime$",
        }

        def mover(inputs, noise):
            return noise

        def obstacle(inputs, noise):
            return noise

        def next_mover_pos(inputs, noise):
            return (
                inputs["mover"] + 1
                if inputs["obstacle"] != inputs["mover"] + 1
                else inputs["mover"]
            )

        self.set_structural_functions(
            {
                "mover": StructuralFunction(
                    mover,
                    [],
                    ExogenousNoise(
                        "u_mover",
                        lambda: np.random.choice(list(range(world_length - 1))),
                    ),
                ),
                "obstacle": StructuralFunction(
                    obstacle,
                    [],
                    ExogenousNoise(
                        "u_obstacle",
                        lambda: np.random.choice(list(range(world_length))),
                    ),
                ),
                "next_mover_pos": StructuralFunction(
                    next_mover_pos, ["mover", "obstacle"], None
                ),
            }
        )
