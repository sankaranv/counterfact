from actual_cause.causal_models.scm import StructuralCausalModel, StructuralFunction
import torch
import pyro.distributions as dist


class BinaryAnd(StructuralCausalModel):
    def __init__(self):
        super().__init__()

        for var in ["a", "b", "y"]:
            self.add_variable(var, "bool", [0, 1])

        def a(inputs, noise):
            return noise

        def b(inputs, noise):
            return noise

        def y(inputs, noise):
            dtype = inputs["a"].dtype
            return torch.logical_and(inputs["a"], inputs["b"]).to(dtype)

        self.set_structural_functions(
            {
                "a": StructuralFunction(a, [], dist.Bernoulli(0.5)),
                "b": StructuralFunction(b, [], dist.Bernoulli(0.5)),
                "y": StructuralFunction(y, ["a", "b"], None),
            }
        )


class BinaryOr(StructuralCausalModel):
    def __init__(self):
        super().__init__()

        for var in ["a", "b", "y"]:
            self.add_variable(var, "bool", [0, 1])

        def a(inputs, noise):
            return noise

        def b(inputs, noise):
            return noise

        def y(inputs, noise):
            dtype = inputs["a"].dtype
            return torch.logical_or(inputs["a"], inputs["b"]).to(dtype)

        self.set_structural_functions(
            {
                "a": StructuralFunction(a, [], dist.Bernoulli(0.5)),
                "b": StructuralFunction(b, [], dist.Bernoulli(0.5)),
                "y": StructuralFunction(y, ["a", "b"], None),
            }
        )


class BinaryXor(StructuralCausalModel):
    def __init__(self):
        super().__init__()

        for var in ["a", "b", "y"]:
            self.add_variable(var, "bool", [0, 1])

        def a(inputs, noise):
            return noise

        def b(inputs, noise):
            return noise

        def y(inputs, noise):
            dtype = inputs["a"].dtype
            return torch.logical_xor(inputs["a"], inputs["b"]).to(dtype)

        self.set_structural_functions(
            {
                "a": StructuralFunction(a, [], dist.Bernoulli(0.5)),
                "b": StructuralFunction(b, [], dist.Bernoulli(0.5)),
                "y": StructuralFunction(y, ["a", "b"], None),
            }
        )
