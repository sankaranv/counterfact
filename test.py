from actual_cause.examples import *


if __name__ == "__main__":

    envs = [
        BinaryAnd(),
        BinaryOr(),
        BinaryXor(),
        ForestFireConjunctive(),
        ForestFireDisjunctive(),
        ForestFireRainStorm(),
        ObedientGang(3),
        HaltOrCharge(),
        SwitchingRailroadTracks(),
        RockThrowing(),
        Voting(5),
    ]
    for env in envs:
        samples = env.sample(10)
        print(samples[0])
