from actual_cause.causal_models.scm import StructuralCausalModel, StructuralFunction
import torch
import pyro.distributions as dist


class HaltOrCharge(StructuralCausalModel):
    def __init__(self):
        super().__init__()

        self.add_variable("major", "discrete", [0, 1, 2])
        self.add_variable("sergeant", "bool", [0, 1])
        self.add_variable("corporal", "bool", [0, 1])

        def major(inputs, noise):
            return noise

        def sergeant(inputs, noise):
            return noise

        def corporal(inputs, noise):
            return torch.where(
                inputs["major"] == 2, inputs["sergeant"], inputs["major"]
            )

        self.set_structural_functions(
            {
                "major": StructuralFunction(
                    major, [], dist.Categorical(torch.tensor([0.5, 0.25, 0.25]))
                ),
                "sergeant": StructuralFunction(sergeant, [], dist.Bernoulli(0.5)),
                "corporal": StructuralFunction(corporal, ["major", "sergeant"], None),
            }
        )
