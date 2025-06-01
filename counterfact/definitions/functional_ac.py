from counterfact.definitions.ac_definition import ACDefinition
from counterfact.definitions.modified_hp import ModifiedHP
from counterfact.causal_models.scm import StructuralCausalModel
import numpy as np
from counterfact.utils.subsets import get_all_subsets


class FunctionalActualCause(ACDefinition):

    def __init__(self, env: StructuralCausalModel):
        super().__init__()
        self.modified_hp = ModifiedHP()
        self.setup_ivp(env)

    def setup_ivp(self, env):
        """
        Set up dictionary to hold IVP assignments
        """

        # Get all variables in the environment
        all_vars = list(env.variables.keys())

        # Collect supports for all variables
        supports = []
        for var_name in all_vars:
            var_support = env.variables[var_name].support
            var_type = env.variables[var_name].var_type
            if var_type == "float":
                raise ValueError(
                    f"Modified HP is not supported for float variable {var_name}"
                )
            elif var_type == "int":
                supports.append(np.arange(var_support[0], var_support[1] + 1, 1))
            else:
                supports.append(var_support)

        # Enumerate all possible states that the environment can generate
        event_combinations = np.array(np.meshgrid(*supports)).T.reshape(
            -1, len(supports)
        )

        # Set up dictionary to hold IVP assignments for each state
        self.state_ivp_assignments = {}
        for state in event_combinations:
            self.ivp_assignments_per_state[tuple(state)] = None

        # Set up dictionary to return all states for a given IVP
        self.ivp_members = {}
        all_subsets = get_all_subsets(all_vars, include_empty=True, include_full=True)
        self.ivp_members = {tuple(subset): set() for subset in all_subsets}

    def set_state_ivp_assignment(self, state, ivp_assignment):
        """
        Set the IVP assignment for a given state
        :param state: Dict
        :param ivp_assignment: Dict
        """
        state_tuple = tuple([state[var] for var in state])
        self.state_ivp_assignments[state_tuple] = ivp_assignment

    def get_state_ivp_assignment(self, state):
        """
        Get the IVP assignment for a given state
        :param state: Dict
        :return: Dict
        """
        state_tuple = tuple([state[var] for var in state])
        return self.state_ivp_assignments[state_tuple]

    def set_ivp_members(self, env, ivp_name, states):
        """
        Set the members of a given IVP
        :param ivp: tuple of state variables
        :param members: Set
        """

        if isinstance(ivp_name, list):
            ivp_key = tuple(ivp_name)
        else:
            ivp_key = ivp_name

        # Ensure the variable names in the tuple are sorted in topological order as defined in the env
        ivp_key = tuple(sorted(ivp_key, key=lambda x: env.topological_order.index(x)))

        if isinstance(states, set):
            self.ivp_members[ivp_key] = states
        elif isinstance(states, list) or isinstance(states, tuple):
            self.ivp_members[ivp_key] = set(states)
        else:
            raise ValueError("Members must be a set or list")

    def get_ivp_members(self, ivp_name):
        """
        Get the members of a given IVP
        :param ivp: tuple of state variables
        :return: Set
        """

        if isinstance(ivp_name, list):
            ivp_name = tuple(ivp_name)
        if ivp_name not in self.ivp_members:
            raise ValueError(f"IVP {ivp_name} not found")
        return self.ivp_members[ivp_name]

    def get_partition_cost(self):
        """
        Get the cost of the set of IVPs
        This is given by sum_i (number of states in a given IVP x number of variables in the key for the IVP)
        :return: float
        """
        return sum(
            [
                len(self.ivp_members[ivp_name]) * len(ivp_name)
                for ivp_name in self.ivp_members
            ]
        )

    def is_necessary(self, event, state, outcome, noise=None):
        """
        Check if the event is a necessary condition for the outcome in the state
        :param event: str
        :param state: Dict[str, Union[float, Callable[[], float]]]
        :param outcome: str
        :param noise: Dict[str, float]
        :return: bool
        """
        return self.modified_hp.is_necessary(self, event, state, outcome, noise)

    def is_sufficient(self, event, state, outcome, noise=None):
        """
        Check if the event is a sufficient condition for the outcome in the state
        :param event: str
        :param state: Dict[str, Union[float, Callable[[], float]]]
        :param outcome: str
        :param noise: Dict[str, float]
        :return: bool
        """

        state_ivp_assignment = self.get_state_ivp_assignment(state)
        if state_ivp_assignment is None:
            return False

        # Check if all variables in the IVP assignment for the given state are in the event
        for var in state_ivp_assignment:
            if var not in event:
                return False

        # Screen out actual causes that fail the Modified HP sufficiency test
        if not self.modified_hp.is_sufficient(self, event, state, outcome, noise):
            return False

        return True

    def is_minimal(self, event, state, outcome, noise=None):
        """
        Check if the event is a minimal condition for the outcome in the state
        :param event: str
        :param state: Dict[str, Union[float, Callable[[], float]]]
        :param outcome: str
        :param noise: Dict[str, float]
        :return: bool
        """
        return True
