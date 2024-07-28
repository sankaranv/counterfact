from actual_cause.causal_models.scm import StructuralCausalModel, StructuralFunction
from actual_cause.causal_models.variables import Variable, ExogenousNoise
import numpy as np


class BinaryAnd(StructuralCausalModel):
    def __init__(self):
        super().__init__()
        self.add_variables(
            [
                Variable("a", "bool"),
                Variable("b", "bool"),
                Variable("y", "bool"),
            ]
        )

        def a(inputs, noise):
            return noise

        def b(inputs, noise):
            return noise

        def y(inputs, noise):
            return inputs["a"] and inputs["b"]

        self.set_structural_functions(
            {
                "a": StructuralFunction(
                    a,
                    [],
                    ExogenousNoise(
                        "u_a", lambda: np.random.choice([0, 1], p=[0.5, 0.5])
                    ),
                ),
                "b": StructuralFunction(
                    b,
                    [],
                    ExogenousNoise(
                        "u_b", lambda: np.random.choice([0, 1], p=[0.5, 0.5])
                    ),
                ),
                "y": StructuralFunction(y, ["a", "b"], None),
            }
        )


class BinaryOr(StructuralCausalModel):
    def __init__(self):
        super().__init__()
        self.add_variables(
            [
                Variable("a", "bool"),
                Variable("b", "bool"),
                Variable("y", "bool"),
            ]
        )

        def a(inputs, noise):
            return noise

        def b(inputs, noise):
            return noise

        def y(inputs, noise):
            return inputs["a"] or inputs["b"]

        self.set_structural_functions(
            {
                "a": StructuralFunction(
                    a,
                    [],
                    ExogenousNoise(
                        "u_a", lambda: np.random.choice([0, 1], p=[0.5, 0.5])
                    ),
                ),
                "b": StructuralFunction(
                    b,
                    [],
                    ExogenousNoise(
                        "u_b", lambda: np.random.choice([0, 1], p=[0.5, 0.5])
                    ),
                ),
                "y": StructuralFunction(y, ["a", "b"], None),
            }
        )


class BinaryXor(StructuralCausalModel):
    def __init__(self):
        super().__init__()
        self.add_variables(
            [
                Variable("a", "bool"),
                Variable("b", "bool"),
                Variable("y", "bool"),
            ]
        )

        def a(inputs, noise):
            return noise

        def b(inputs, noise):
            return noise

        def y(inputs, noise):
            return (inputs["a"] and not inputs["b"]) or (
                not inputs["a"] and inputs["b"]
            )

        self.set_structural_functions(
            {
                "a": StructuralFunction(
                    a,
                    [],
                    ExogenousNoise(
                        "u_a", lambda: np.random.choice([0, 1], p=[0.5, 0.5])
                    ),
                ),
                "b": StructuralFunction(
                    b,
                    [],
                    ExogenousNoise(
                        "u_b", lambda: np.random.choice([0, 1], p=[0.5, 0.5])
                    ),
                ),
                "y": StructuralFunction(y, ["a", "b"], None),
            }
        )
