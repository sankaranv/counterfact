from counterfact.causal_models.scm import StructuralCausalModel, StructuralFunction
from counterfact.causal_models.variables import Variable, ExogenousNoise
import numpy as np


class ObedientGang(StructuralCausalModel):
    def __init__(self, n_members=3):
        super().__init__()
        self.add_variables(
            [Variable(f"leader", "bool"), Variable(f"death", "bool")]
            + [Variable(f"gang_member_{i}", "bool") for i in range(n_members)]
        )

        def leader(inputs, noise):
            return noise

        def gang_member(inputs, noise):
            return inputs["leader"]

        def death(inputs, noise):
            return int(
                np.logical_or.reduce(
                    [inputs[f"gang_member_{i}"] for i in range(n_members)]
                    + [inputs["leader"]]
                )
            )

        self.set_structural_functions(
            {
                "leader": StructuralFunction(
                    leader,
                    [],
                    ExogenousNoise(
                        "u_leader", lambda: np.random.choice([0, 1], p=[0.5, 0.5])
                    ),
                ),
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
