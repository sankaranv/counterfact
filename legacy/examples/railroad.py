import pymc as pm
from pymc.math import switch, eq
from actual_cause.environment import Environment


class SwitchingRailroadTracks(Environment):
    def __init__(self, n_samples_per_obs=1):
        super().__init__(n_samples_per_obs)

    def build_model(self):

        self.description = """A railway engineer is in charge of switching the incoming train to the left or right
        track. The values of the switch variable indicates whether the left or right track was chosen, and the track
        variable indicates what track the train is on. If there is a breakdown, indicated by track = 2, the train will
        not arrive at the station, but if it is on the left or right track it will arrive at the station.
        """

        with self.model:

            # Exogenous variables as priors
            u_breakdown = pm.Beta("u_breakdown", 1, 1)
            u_track_switcher = pm.Beta("u_track_switcher", 1, 1)

            # Track is set to 2 if there is a breakdown, 0 indicates left track, 1 indicates right track
            breakdown = pm.Bernoulli("breakdown", u_breakdown)
            track_switcher = pm.Bernoulli("track_switcher", u_track_switcher)
            on_track = pm.Deterministic(
                "on_track", switch(eq(breakdown, 1), 2, track_switcher)
            )

            # Arrived is set to 1 if the train did not break down
            arrived = pm.Deterministic("arrived", switch(eq(on_track, 2), 0, 1))
