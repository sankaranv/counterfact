import os
import pandas as pd


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

    # Use the order of variables in the columns of the state variables
    var_order = [var_names[col[1]] for col in df.columns if col[0] == "state"]

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
        for i, causes in enumerate(formatted_df[("Actual Causes", "")]):

            for j, cause in enumerate(causes):
                if len(cause) == 1:
                    causes[j] = var_names[cause[0]]
                else:
                    causes[j] = [var_names[var] for var in cause]
                    print(causes[j])
                    # Sort using the var_order
                    causes[j] = sorted(
                        causes[j],
                        key=lambda x: (
                            var_order.index(x) if len(x) > 1 else var_order.index(x[0])
                        ),
                    )

            # Sort causes based on the first element of each cause, or the cause itself if it is a singleton
            causes = sorted(
                causes,
                key=lambda x: (
                    var_order.index(x) if len(x) > 1 else var_order.index(x[0])
                ),
            )
            formatted_df.at[i, ("Actual Causes", "")] = ", ".join(causes)

    # Wrap all numeric values in math mode
    formatted_df = formatted_df.map(to_latex_math_mode)

    # Align the last column to the left if there are human-readable variable names as actual causes, otherwise center
    if ("Actual Causes", "") in formatted_df.columns:
        last_col_align = "l"
    else:
        last_col_align = "c"
    column_format = "c" * (len(formatted_df.columns) - 1) + last_col_align

    print(formatted_df)

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

    # Check if filename already ends with .tex
    if filename.endswith(".tex"):
        with open(filename, "w") as f:
            f.write(latex_table)
    else:
        with open(filename + ".tex", "w") as f:
            f.write(latex_table)
