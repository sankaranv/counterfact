from actual_cause.causal_models.scm import StructuralCausalModel, StructuralFunction
from actual_cause.causal_models.variables import Variable, ExogenousNoise
import numpy as np


description = """
            Billy and Suzy are running for election and there are n voters. The outcome is a binary variable winner where 1 
            indicates that Suzy wins and 0 indicates that Billy wins. There are n binary variables indicating each 
            person's vote, one variable per voter. A candidate needs to have the majority in order to win.
            """


class Voting(StructuralCausalModel):

    def __init__(self, n_voters=11):
        super().__init__()
        self.add_variables(
            [Variable(f"voter_{i}", "bool") for i in range(1, n_voters + 1)]
            + [Variable("winner", "bool")]
        )

        def voter(inputs, noise):
            return noise

        def winner(inputs, noise):
            return int(
                sum([inputs[f"voter_{i}"] for i in range(1, n_voters + 1)])
                >= n_voters / 2
            )

        self.set_structural_functions(
            {
                f"voter_{i}": StructuralFunction(
                    voter,
                    [],
                    ExogenousNoise(
                        f"u_voter_{i}", lambda: np.random.choice([0, 1], p=[0.5, 0.5])
                    ),
                )
                for i in range(1, n_voters + 1)
            }
            | {
                "winner": StructuralFunction(
                    winner,
                    parents=[f"voter_{i}" for i in range(1, n_voters + 1)],
                    noise_dist=None,
                )
            }
        )
