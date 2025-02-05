from actual_cause.causal_models.scm import StructuralCausalModel, StructuralFunction
import torch
import pyro.distributions as dist

# next_mover_pos = mover + 1 if obstacle != mover + 1 else mover


class Mover1D(StructuralCausalModel):
    def __init__(self, world_length: int = 4):
        super().__init__()

        self.add_variable("mover", "int", [0, world_length - 1])
        self.add_variable("obstacle", "int", [0, world_length])
        self.add_variable("next_mover_pos", "int", [1, world_length])

        self.formatted_var_names = {
            "mover": "$m$",
            "obstacle": "$o$",
            "next_mover_pos": "$m^'$",
        }

        def mover(inputs, noise):
            return noise

        def obstacle(inputs, noise):
            return noise

        def next_mover_pos(inputs, noise):
            return torch.where(
                inputs["obstacle"] != inputs["mover"] + 1,
                inputs["mover"] + 1,
                inputs["mover"],
            )

        self.set_structural_functions(
            {
                "mover": StructuralFunction(
                    mover,
                    [],
                    dist.Categorical(torch.tensor([1 / world_length] * world_length)),
                ),
                "obstacle": StructuralFunction(
                    obstacle,
                    [],
                    dist.Categorical(
                        torch.tensor([1 / (world_length + 1)] * (world_length + 1))
                    ),
                ),
                "next_mover_pos": StructuralFunction(
                    next_mover_pos, ["mover", "obstacle"], None
                ),
            }
        )
