from counterfact.causal_models.scm import StructuralCausalModel, StructuralFunction
from counterfact.causal_models.variables import Variable, ExogenousNoise
import numpy as np


class SwitchingRailroadTracks(StructuralCausalModel):

    def __init__(self):
        super().__init__()
        self.add_variables(
            [
                Variable("breakdown", "bool"),
                Variable("track_switcher", "bool"),
                Variable("on_track", "bool"),
                Variable("arrived", "bool"),
            ]
        )

        def breakdown(inputs, noise):
            return noise

        def track_switcher(inputs, noise):
            return noise

        def on_track(inputs, noise):
            return 2 if inputs["breakdown"] else inputs["track_switcher"]

        def arrived(inputs, noise):
            return 1 if not inputs["breakdown"] else 0

        self.set_structural_functions(
            {
                "breakdown": StructuralFunction(
                    breakdown,
                    [],
                    ExogenousNoise(
                        "u_breakdown", lambda: np.random.choice([0, 1], p=[0.5, 0.5])
                    ),
                ),
                "track_switcher": StructuralFunction(
                    track_switcher,
                    [],
                    ExogenousNoise(
                        "u_track_switcher",
                        lambda: np.random.choice([0, 1], p=[0.5, 0.5]),
                    ),
                ),
                "on_track": StructuralFunction(
                    on_track, ["breakdown", "track_switcher"], None
                ),
                "arrived": StructuralFunction(arrived, ["breakdown"], None),
            }
        )
