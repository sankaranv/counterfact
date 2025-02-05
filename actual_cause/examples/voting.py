from actual_cause.causal_models.scm import StructuralCausalModel, StructuralFunction
import torch
import pyro.distributions as dist


class Voting(StructuralCausalModel):

    def __init__(self, n_voters=11):
        super().__init__()

        # Add all variables
        for var in [f"voter_{i}" for i in range(1, n_voters + 1)] + ["winner"]:
            self.add_variable(var, "bool", [0, 1])

        def voter(inputs, noise):
            return noise

        def winner(inputs, noise):
            voter_results = torch.Tensor(
                [inputs[f"voter_{i}"] for i in range(1, n_voters + 1)]
            )
            dtype = voter_results[0].dtype
            return (torch.sum(voter_results) > n_voters / 2).to(dtype)

        self.set_structural_functions(
            {
                f"voter_{i}": StructuralFunction(voter, [], dist.Bernoulli(0.5))
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
