class BinarySAT:

    def __init__(self, env, ac_defn):

        # Check if all variables are binary, otherwise raise an error
        for var in env.variables:
            if env.variables[var].type != "bool":
                raise ValueError(f"Variable {var} is not binary, cannot use SAT solver")

        self.env = env
        self.ac_defn = ac_defn

    def find_witness_set(self):

        pass
