from actual_cause.causal_models.scm import StructuralCausalModel, StructuralFunction
import torch
import pyro.distributions as dist


class ObedientGang(StructuralCausalModel):
    def __init__(self, n_members=3):
        super().__init__()

        for var in [f"gang_member_{i}" for i in range(n_members)] + ["leader", "death"]:
            self.add_variable(var, "bool", [0, 1])

        def leader(inputs, noise):
            return noise

        def gang_member(inputs, noise):
            return inputs["leader"]

        def death(inputs, noise):
            dtype = inputs["leader"].dtype
            return torch.any(
                torch.stack(
                    [inputs[f"gang_member_{i}"] for i in range(n_members)]
                    + [inputs["leader"]]
                ),
                dim=0,
            ).to(dtype)

        self.set_structural_functions(
            {
                "leader": StructuralFunction(leader, [], dist.Bernoulli(0.5)),
                **{
                    f"gang_member_{i}": StructuralFunction(
                        gang_member, ["leader"], None
                    )
                    for i in range(n_members)
                },
                "death": StructuralFunction(
                    death,
                    [f"gang_member_{i}" for i in range(n_members)] + ["leader"],
                    None,
                ),
            }
        )
