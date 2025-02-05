from actual_cause.causal_models.scm import StructuralCausalModel, StructuralFunction
import torch
import pyro.distributions as dist


class QueenOfEngland(StructuralCausalModel):
    def __init__(self):
        super().__init__()

        self.add_variable("queen", "discrete", [0, 1, 2])
        self.add_variable("gardener", "bool", [0, 1])
        self.add_variable("flowers_live", "bool", [0, 1])

        def queen(inputs, noise):
            return noise

        def gardener(inputs, noise):
            return noise

        def flowers_live(inputs, noise):
            dtype = inputs["queen"].dtype
            return torch.logical_or(
                inputs["gardener"], torch.eq(inputs["queen"], 1)
            ).to(dtype)

        self.set_structural_functions(
            {
                "queen": StructuralFunction(
                    queen, [], dist.Categorical(torch.Tensor([0.1, 0.8, 0.1]))
                ),
                "gardener": StructuralFunction(gardener, [], dist.Bernoulli(0.5)),
                "flowers_live": StructuralFunction(
                    flowers_live, ["queen", "gardener"], None
                ),
            }
        )
