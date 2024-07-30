from actual_cause.definitions import ModifiedHP
from actual_cause.examples.mover_1d import Mover1D

env = Mover1D(world_length=4)
ac_defn = ModifiedHP()
state = {"mover": 0, "obstacle": 1, "next_mover_pos": 1}
event = {"obstacle": 1}
outcome = {"next_mover_pos": 1}
noise = {"mover": 0, "obstacle": 2}
factual, info = ac_defn.is_factual(env, event, outcome, state, noise)
print(f"Factual: {factual} Info: {info}")
sufficient, info = ac_defn.is_sufficient(env, event, outcome, state, noise)
print(f"Sufficient: {sufficient} Info: {info}")
necessary, info = ac_defn.is_necessary(env, event, outcome, state, noise)
print(f"Necessary: {necessary} Info: {info}")
minimal, info = ac_defn.is_minimal(env, event, outcome, state, noise)
print(f"Minimal: {minimal} Info: {info}")
result, info = ac_defn.is_actual_cause(env, event, outcome, state, noise)
print(f"Actual Cause: {result}! Info: {info}")
