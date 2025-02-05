from actual_cause.causal_models.scm import StructuralCausalModel, StructuralFunction
import torch
import pyro.distributions as dist


class SwitchingRailroadTracks(StructuralCausalModel):

    def __init__(self):
        super().__init__()

        # Add all variables
        for var in ["breakdown", "track_switcher", "on_track", "arrived"]:
            self.add_variable(var, "bool", [0, 1])

        def breakdown(inputs, noise):
            return noise["breakdown"]

        def track_switcher(inputs, noise):
            return noise["track_switcher"]

        def on_track(inputs, noise):
            return (
                torch.Tensor(2.0) if inputs["breakdown"] else inputs["track_switcher"]
            )

        def arrived(inputs, noise):
            return torch.Tensor(1.0) if not inputs["breakdown"] else torch.Tensor(0.0)

        self.set_structural_functions(
            {
                "breakdown": StructuralFunction(breakdown, [], dist.Bernoulli(0.1)),
                "track_switcher": StructuralFunction(
                    track_switcher, [], dist.Bernoulli(0.5)
                ),
                "on_track": StructuralFunction(
                    on_track, ["breakdown", "track_switcher"], None
                ),
                "arrived": StructuralFunction(arrived, ["breakdown"], None),
            }
        )
