import dataclasses

@dataclasses.dataclass
class Continuous:
    low: float
    high: float

    def __contains__(self, value):
        return self.low <= value <= self.high

    def __repr__(self):
        return f"[{self.low}, {self.high}]"

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        return self.low == other.low and self.high == other.high

    def __ne__(self, other):
        return not self.__eq__(other)

@dataclasses.dataclass
class Discrete:
    def __init__(self, values):
        self.values = set(values)

    def __contains__(self, value):
        return value in self.values

    def __repr__(self):
        return f"{self.values}"

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        return self.values == other.values

    def __ne__(self, other):
        return not self.__eq__(other)


@dataclasses.dataclass
class Integer:
    low: int
    high: int

    def __contains__(self, value):
        if self.low <= value <= self.high and value == int(value):
            return True

    def __repr__(self):
        return f"[{self.low}, {self.high}]"

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        return self.low == other.low and self.high == other.high

    def __ne__(self, other):
        return not self.__eq__(other)

@dataclasses.dataclass
class Boolean:
    def __contains__(self, value):
        return value in [True, False]

    def __repr__(self):
        return f"[True, False]"

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return not self.__eq__(other)


