from actual_cause.definitions import ModifiedHP
from actual_cause.examples.mover_1d import Mover1D
from actual_cause.inference import get_all_actual_causes
from actual_cause.utils.utils import *

env = Mover1D(world_length=4)
ac_defn = ModifiedHP()
# state = {"mover": 0, "obstacle": 1, "next_mover_pos": 1}
# event = {"obstacle": 1}
# outcome = {"next_mover_pos": 1}
# noise = {"mover": 0, "obstacle": 2}
# factual, info = ac_defn.is_factual(env, event, outcome, state, noise)
# print(f"Factual: {factual} Info: {info}")
# sufficient, info = ac_defn.is_sufficient(env, event, outcome, state, noise)
# print(f"Sufficient: {sufficient} Info: {info}")
# necessary, info = ac_defn.is_necessary(env, event, outcome, state, noise)
# print(f"Necessary: {necessary} Info: {info}")
# minimal, info = ac_defn.is_minimal(env, event, outcome, state, noise)
# print(f"Minimal: {minimal} Info: {info}")
# result, info = ac_defn.is_actual_cause(env, event, outcome, state, noise)
# print(f"Actual Cause: {result}! Info: {info}")

ac_table = get_all_actual_causes(env, ac_defn, ["next_mover_pos"])
print(ac_table)

make_latex_table(ac_table, "tables/mover_1d", env.formatted_var_names)
