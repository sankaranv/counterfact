import pymc as pm
from pymc.math import switch, sum, ge
from actual_cause.environment import Environment


class Voting(Environment):

    def __init__(self, n_samples_per_obs=1):
        super().__init__(n_samples_per_obs=n_samples_per_obs)

    def build_model(self):

        self.description = """
            Billy and Suzy are running for election and there are 11 voters. The outcome is a binary variable where 1 
            indicates that Suzy wins and 0 indicates that Billy wins. There are 11 binary variables indicating each 
            person's vote. A candidate needs to have the majority, or at least 6 votes, in order to win.
            """
        with self.model:
            u_voter_1 = pm.Beta("u_voter_1", alpha=1, beta=1)
            u_voter_2 = pm.Beta("u_voter_2", alpha=1, beta=1)
            u_voter_3 = pm.Beta("u_voter_3", alpha=1, beta=1)
            u_voter_4 = pm.Beta("u_voter_4", alpha=1, beta=1)
            u_voter_5 = pm.Beta("u_voter_5", alpha=1, beta=1)
            u_voter_6 = pm.Beta("u_voter_6", alpha=1, beta=1)
            u_voter_7 = pm.Beta("u_voter_7", alpha=1, beta=1)
            u_voter_8 = pm.Beta("u_voter_8", alpha=1, beta=1)
            u_voter_9 = pm.Beta("u_voter_9", alpha=1, beta=1)
            u_voter_10 = pm.Beta("u_voter_10", alpha=1, beta=1)
            u_voter_11 = pm.Beta("u_voter_11", alpha=1, beta=1)

            voter_1 = pm.Bernoulli("voter_1", p=u_voter_1)
            voter_2 = pm.Bernoulli("voter_2", p=u_voter_2)
            voter_3 = pm.Bernoulli("voter_3", p=u_voter_3)
            voter_4 = pm.Bernoulli("voter_4", p=u_voter_4)
            voter_5 = pm.Bernoulli("voter_5", p=u_voter_5)
            voter_6 = pm.Bernoulli("voter_6", p=u_voter_6)
            voter_7 = pm.Bernoulli("voter_7", p=u_voter_7)
            voter_8 = pm.Bernoulli("voter_8", p=u_voter_8)
            voter_9 = pm.Bernoulli("voter_9", p=u_voter_9)
            voter_10 = pm.Bernoulli("voter_10", p=u_voter_10)
            voter_11 = pm.Bernoulli("voter_11", p=u_voter_11)

            # The final vote is the sum of all the voters
            final_vote = sum(
                [
                    voter_1,
                    voter_2,
                    voter_3,
                    voter_4,
                    voter_5,
                    voter_6,
                    voter_7,
                    voter_8,
                    voter_9,
                    voter_10,
                    voter_11,
                ]
            )
            result = pm.Deterministic("result", switch(ge(final_vote, 6), 1, 0))
