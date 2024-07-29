from actual_cause.causal_models.scm import StructuralCausalModel
from actual_cause.utils.utils import *


class ACDefinition:

    def __init__(self):
        pass

    def is_factual(
        self,
        env: StructuralCausalModel,
        event: dict,
        outcome: dict,
        state: dict,
        noise=None,
    ):
        """
        Check if the event and outcome actually occurred in the given state
        This is AC1 in the definition of actual cause
        :param env: StructuralCausalModel, will not be used for most deterministic definitions
        :param event: dictionary of values of a given set of variables
        :param state: dictionary of values of all observable variables
        :param outcome: dictionary of values of the outcome variables
        :param noise: dictionary of values of all exogenous noise variables
        :return:
        """

        for var, value in event.items():
            if state[var] != value:
                return False
        for var, value in outcome.items():
            if state[var] != value:
                return False
        return True

    def is_necessary(
        self,
        env: StructuralCausalModel,
        event: dict,
        outcome: dict,
        state: dict,
        noise=None,
    ):
        """
        Check if the event is a necessary condition for the outcome obtained in the given state
        :param env: StructuralCausalModel
        :param event: dictionary of values of a given set of variables
        :param outcome: dictionary of values of the outcome variables
        :param state: dictionary of values of all observable variables
        :param noise: dictionary of values of all exogenous noise variables
        :return:
        """
        raise NotImplementedError("Necessity condition not implemented")

    def is_sufficient(
        self,
        env: StructuralCausalModel,
        event: dict,
        outcome: dict,
        state: dict,
        noise=None,
    ):
        """
        Check if the event is a sufficient condition for the outcome obtained in the given state
        :param env: StructuralCausalModel
        :param event: dictionary of values of a given set of variables
        :param outcome: dictionary of values of the outcome variables
        :param state: dictionary of values of all observable variables
        :param noise: dictionary of values of all exogenous noise variables
        :return:
        """
        raise NotImplementedError("Sufficiency condition not implemented")

    def is_minimal(
        self,
        env: StructuralCausalModel,
        event: dict,
        outcome: dict,
        state: dict,
        noise=None,
    ):
        """
        Check if a given actual cause is minimal
        This is AC3 in the definition of actual cause
        :param env:
        :param event:
        :param outcome:
        :param state:
        :param noise:
        :return:
        """

        # Base case: singleton event
        if len(event.keys()) == 1:
            return True, "Event is a minimal actual cause"

        # Find all possible subsets of the set of variables in the event
        for subevent in get_all_subevents(event, reverse=True):
            env.reset()
            subevent_is_necessary = self.is_necessary(
                env, subevent, outcome, state, noise
            )
            subevent_is_sufficient = self.is_sufficient(
                env, subevent, outcome, state, noise
            )
            if subevent_is_necessary and subevent_is_sufficient:
                return (
                    False,
                    f"Not minimal, {subevent} is also necessary and sufficient",
                )

        return True, "Event is a minimal actual cause"

    def is_actual_cause(self, env, event, outcome, state, noise=None):
        """
        Check if the event is an actual cause of the outcome in the state
        Conditions are ordered by increasing computational cost
        :param env: StructuralCausalModel
        :param event: dictionary of values of a given set of variables
        :param outcome: dictionary of values of the outcome variables
        :param state: dictionary of values of all observable variables
        :param noise: dictionary of values of all exogenous noise variables
        :return: answer: bool indicating whether the event is an actual cause
        :return: info: str with additional info if any criteria were failed
        """

        # TODO - incorporate the witness set

        # Check for AC1
        if not self.is_factual(env, event, outcome, state, noise):
            return False, "Failed AC1"

        # Check for AC2b
        if not self.is_sufficient(env, event, outcome, state, noise):
            return False, "Not sufficient"

        # Check for AC2a
        if not self.is_necessary(env, event, outcome, state, noise):
            return False, "Not necessary"

        # Check for AC3
        minimal, info = self.is_minimal(env, event, outcome, state, noise)
        if not minimal:
            return False, info

        return True, info

    def get_actual_causes(self, env, outcome, state, noise=None):
        """
        Find all actual causes of the outcome in the state
        :param env: StructuralCausalModel
        :param outcome: dictionary of values of the outcome variables
        :param state: dictionary of values of all observable variables
        :param noise: dictionary of values of all exogenous noise variables
        :return:
        """
        pass
