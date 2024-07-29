def rock_throwing_hp_test():
    env = RockThrowing()
    ac_defn = ModifiedHP()
    state = {
        "suzy_throws": 1,
        "billy_throws": 1,
        "suzy_hits": 1,
        "billy_hits": 0,
        "bottle_shatters": 1,
    }
    event = {"suzy_throws": 1}
    outcome = {"bottle_shatters": 0}
    print(f"Event: {event}")
    print(f"Outcome: {outcome}")
    result, info = ac_defn.is_actual_cause(
        env, event, outcome, state, noise={"suzy_throws": 1, "billy_throws": 1}
    )
    print(result, info)


if __name__ == "__main__":
    rock_throwing_hp_test()
