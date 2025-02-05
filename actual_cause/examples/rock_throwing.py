from actual_cause.causal_models.scm import StructuralCausalModel, StructuralFunction
import torch
import pyro.distributions as dist


class RockThrowing(StructuralCausalModel):

    # TODO - update so that the parents correspond to the causal graph

    def __init__(self):
        super().__init__()

        # Add all variables

        for var in [
            "suzy_throws",
            "billy_throws",
            "suzy_hits",
            "billy_hits",
            "bottle_shatters",
        ]:
            self.add_variable(var, "bool", [0, 1])

        # Create structural functions
        def suzy_throws(inputs, noise):
            return noise["suzy_throws"]

        def billy_throws(inputs, noise):
            return noise["billy_throws"]

        def suzy_hits(inputs, noise):
            return inputs["suzy_throws"]

        def billy_hits(inputs, noise):
            return inputs["billy_throws"] * (1 - inputs["suzy_hits"])

        def bottle_shatters(inputs, noise):
            dtype = inputs["suzy_hits"].dtype
            return torch.logical_or(inputs["suzy_hits"], inputs["billy_hits"]).to(dtype)

        # Set structural functions for the model

        self.set_structural_functions(
            {
                "suzy_throws": StructuralFunction(suzy_throws, [], dist.Bernoulli(0.5)),
                "billy_throws": StructuralFunction(
                    billy_throws, [], dist.Bernoulli(0.5)
                ),
                "suzy_hits": StructuralFunction(suzy_hits, ["suzy_throws"], None),
                "billy_hits": StructuralFunction(
                    billy_hits, ["billy_throws", "suzy_hits"], None
                ),
                "bottle_shatters": StructuralFunction(
                    bottle_shatters, ["suzy_hits", "billy_hits"], None
                ),
            }
        )
