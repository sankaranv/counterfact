from typing import List, Union, Callable


class Variable:
    def __init__(
        self,
        name: str,
        var_type: str,
        support: List[Union[int, float]] = None,
    ):

        if support is None:
            support = [0, 1]
        if var_type not in ["int", "discrete", "float", "bool"]:
            raise ValueError(
                f"Invalid type {var_type} for variable {name}, should be int, discrete, float, or bool."
            )

        self.name = name
        self.var_type = var_type
        self.support = support
        self.validate_support()

    def validate_support(self):
        if self.var_type == "bool":
            if not isinstance(self.support, list) or len(self.support) != 2:
                raise ValueError(
                    f"Support for boolean variable '{self.name}' must be list of two numbers."
                )
        elif self.var_type == "int":
            if (
                not isinstance(self.support, list)
                or len(self.support) != 2
                or not all(isinstance(x, int) for x in self.support)
            ):
                raise ValueError(
                    f"Support for integer variable '{self.name}' must be a list of two integers."
                )
        elif self.var_type == "float":
            if (
                not isinstance(self.support, list)
                or len(self.support) != 2
                or not all(isinstance(x, (int, float)) for x in self.support)
            ):
                raise ValueError(
                    f"Support for float variable '{self.name}' must be a list of two numbers."
                )
        elif self.var_type == "discrete":
            if not isinstance(self.support, list) or len(self.support) == 0:
                raise ValueError(
                    f"Support for discrete variable '{self.name}' must be a non-empty list of values."
                )
        else:
            raise ValueError(
                f"Unsupported variable type '{self.var_type}' for variable '{self.name}'."
            )


# Class for Exogenous Noise
class ExogenousNoise:
    def __init__(self, name: str, distribution: Callable[[], float]):
        self.name = name
        self.distribution = distribution

    def sample(self) -> float:
        return self.distribution()
