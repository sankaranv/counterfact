import pymc as pm
from pymc.math import switch
from actual_cause.environment import Environment


class ObedientGang(Environment):

    def __init__(self, n_samples_per_obs=1):
        super().__init__(n_samples_per_obs=n_samples_per_obs)

    def build_model(self):

        self.description = """
        There are three members in a gang who will always obey the leader's command. The leader
        decides whether to shoot or not, and if the leader makes the order then all gang members
        will shoot"
        """

        with self.model:
            u_leader = pm.Beta("u_leader", 1, 1)
            leader = pm.Bernoulli("leader", u_leader)

            # The gang members will follow the leader
            gang_member_1 = pm.Deterministic("gang_member_1", leader)
            gang_member_2 = pm.Deterministic("gang_member_2", leader)
            gang_member_3 = pm.Deterministic("gang_member_3", leader)

            # The victim will die if any of them shoot
            death = pm.Deterministic(
                "death", switch(gang_member_1 + gang_member_2 + gang_member_3 > 0, 1, 0)
            )
