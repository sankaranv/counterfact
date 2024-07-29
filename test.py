from actual_cause.examples import RockThrowing
from actual_cause.definitions import ModifiedHP

env = RockThrowing()
ac_defn = ModifiedHP()
state = {
    "suzy_throws": 1,
    "billy_throws": 1,
    "suzy_hits": 1,
    "billy_hits": 0,
    "bottle_shatters": 1,
}
event = {"suzy_throws": 1, "billy_hits": 0}
outcome = {"bottle_shatters": 1}
noise = {"suzy_throws": 1, "billy_throws": 1}
print(f"Event: {event}")
print(f"Outcome: {outcome}")
print(f"State: {state}")
result, info = ac_defn.is_sufficient(env, event, outcome, state, noise)
print(f"Sufficient: {result}")
print(f"Sufficiency info: {info}")
result, info = ac_defn.is_necessary(env, event, outcome, state, noise)
print(f"Necessary: {result}")
print(f"Necessity info: {info}")
result, info = ac_defn.is_minimal(env, event, outcome, state, noise)
print(f"Minimal: {result}")
print(f"Minimality info: {info}")
print("ac3_smaller_cause" not in info)
