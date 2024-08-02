from itertools import chain, combinations
import os
import pandas as pd


def powerset(iterable, reverse=False):
    s = list(iterable)
    if reverse:
        return chain.from_iterable(combinations(s, r) for r in range(len(s) - 1, 0, -1))
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)))


def get_all_subevents(event, reverse=False):
    vars = event.keys()
    event_combinations = list(powerset(vars, reverse))
    subevents = [{k: event[k] for k in subevent} for subevent in event_combinations]
    return subevents


def add_info(info, updates):
    """
    Add elements of second dictionary to first dictionary
    If any keys from the updates dict are already in info, the values are overwritten
    :param info: dict
    :param updates: dict
    :return: combined dict
    """
    for k, v in updates.items():
        info[k] = v
    return info


def to_latex_math_mode(value):
    if isinstance(value, (int, float)):
        return f"${value}$"
    return value


def make_latex_table(df: pd.DataFrame, filename: str, var_names: dict):
    """
    Make a formatted latex table from a dataframe and save it to file
    :param df: pd.DataFrame containing states, outcomes, and actual causes
    :param filename: output file name
    :param var_names: dict mapping variable names to human-readable names
    :return:
    """

    # Get the number of state and outcome variables
    num_state_vars = sum([1 for col in df.columns if col[0] == "state"])
    num_outcome_vars = sum([1 for col in df.columns if col[0] == "outcome"])

    # Columns in dataframe are either ('state, 'old_var_name'), ('outcome', 'old_var_name'), or ('actual_causes', '')
    # Rename to ('State', 'new_var_name'), ('Outcome', 'new_var_name'), ('Actual Causes', '')
    columns = []
    for col in df.columns:
        if col[0] == "state":
            columns.append(("State", var_names[col[1]]))
        elif col[0] == "outcome":
            columns.append(("Outcome", var_names[col[1]]))
        elif col[0] == "actual_causes":
            columns.append(("Actual Causes", ""))
        elif col[0] == "noise":
            columns.append(("Noise", var_names[col[1]]))
        elif col[0] == "binary":
            columns.append(("Binary", "$\\mathbf{b}$"))
        else:
            columns.append(col)

    # Make new dataframe with remapped columns, don't mess with old dataframe
    formatted_df = pd.DataFrame(df.values, columns=pd.MultiIndex.from_tuples(columns))

    # Actual Causes column contains a list of variables names, remap them to human-readable form and join into a string
    if ("Actual Causes", "") in formatted_df.columns:
        formatted_df[("Actual Causes", "")] = formatted_df[("Actual Causes", "")].apply(
            lambda x: ", ".join([var_names[var] for var in x])
        )

    # Wrap all numeric values in math mode
    formatted_df = formatted_df.map(to_latex_math_mode)

    # Align the last column to the left if there are human-readable variable names as actual causes, otherwise center
    if ("Actual Causes", "") in formatted_df.columns:
        last_col_align = "l"
    else:
        last_col_align = "c"
    column_format = "c" * (len(formatted_df.columns) - 1) + last_col_align

    # Generate latex code
    latex_table = formatted_df.to_latex(
        index=False, column_format=column_format, multicolumn_format="c"
    )

    # Add cmidrule under multicols to separate state, outcome, and actual causes
    for i, line in enumerate(latex_table.split("\n")):
        if "State" in line and "Outcome" in line:
            if num_state_vars > 1:
                # Multiple state and outcome variables
                if num_outcome_vars > 1:
                    latex_table = latex_table.replace(
                        line,
                        line
                        + " \\cmidrule(lr){1-"
                        + str(num_state_vars)
                        + "} \\cmidrule(lr){"
                        + str(num_state_vars + 1)
                        + "-"
                        + str(num_state_vars + num_outcome_vars),
                    )
                # Multiple state variables, single outcome variable - only cmidrule for states
                else:
                    latex_table = latex_table.replace(
                        line,
                        line + " \\cmidrule(lr){1-" + str(num_state_vars) + "}",
                    )
            else:
                # Single state variable, multiple outcome variables - only cmidrule for outcomes
                if num_outcome_vars > 1:
                    latex_table = latex_table.replace(
                        line,
                        line + " \\cmidrule(lr){" + str(num_state_vars + 1) + "-",
                    )

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename + ".tex", "w") as f:
        f.write(latex_table)
