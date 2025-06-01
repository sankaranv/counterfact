from counterfact.definitions import (
    ModifiedHP,
    DirectActualCause,
    FunctionalActualCause,
)
from counterfact.examples import RockThrowing
from counterfact.inference import HPExhaustiveSearch
from counterfact.utils import *
import sys

env = RockThrowing()

ac_defn = FunctionalActualCause()
sys.exit(0)
ac_defn = DirectActualCause()
state = {
    "suzy_throws": 1,
    "billy_throws": 1,
    "suzy_hits": 1,
    "billy_hits": 0,
    "bottle_shatters": 1,
}
noise = {"suzy_throws": 1, "billy_throws": 1}
event = {"suzy_throws": 1}
outcome = {"bottle_shatters": 1}

# factual, info = ac_defn.is_factual(env, event, outcome, state, noise)
# print(f"Factual: {factual} Info: {info}")
sufficient, info = ac_defn.is_sufficient(
    env, event, outcome, state, noise, filter_non_parents=True
)
print(f"Sufficient: {sufficient} Info: {info}")
# necessary, info = ac_defn.is_necessary(env, event, outcome, state, noise)
# print(f"Necessary: {necessary} Info: {info}")
# minimal, info = ac_defn.is_minimal(env, event, outcome, state, noise)
# print(f"Minimal: {minimal} Info: {info}")
# result, info = ac_defn.is_actual_cause(env, event, outcome, state, noise)
# print(f"Actual Cause: {result}! Info: {info}")

# print("\n Running solver...\n")

# solver = HPExhaustiveSearch(env, ac_defn)
# ac_table = solver.solve_all_states(env, ac_defn, outcome_vars=["bottle_shatters"])
# var_names = {
#     "suzy_throws": "ST",
#     "billy_throws": "BT",
#     "suzy_hits": "SH",
#     "billy_hits": "BH",
#     "bottle_shatters": "BS",
# }
#
# make_latex_table(ac_table, "tables/rock_throwing_table.tex", var_names)

# actual_causes = solver.solve(state, outcome, noise)
# print(actual_causes.keys())
