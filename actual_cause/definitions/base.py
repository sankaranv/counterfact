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
        **kwargs,
    ):
        """
        Check if the event and outcome actually occurred in the given state
        This is AC1 in the definition of actual cause
        :param env: StructuralCausalModel, will not be used for most deterministic definitions
        :param event: dictionary of values of a given set of variables
        :param state: dictionary of values of all observable variables
        :param outcome: dictionary of values of the outcome variables
        :param noise: dictionary of values of all exogenous noise variables
        :param kwargs: additional arguments needed for the particular definition
        :return: answer: bool indicating whether the event and outcome are factual
        :return: info: dict with additional info about the factuality test
        """
        info = {}
        factual = True
        incorrect_events = {}
        incorrect_outcomes = {}

        # Check if the event and outcome are factual
        for var, value in event.items():
            if state[var] != value:
                incorrect_events[var] = event[var]
                factual = False
        for var, value in outcome.items():
            if state[var] != value:
                incorrect_outcomes[var] = event[var]
                factual = False

        # Report any incorrect events or outcomes
        if incorrect_events:
            info["incorrect_events"] = incorrect_events
        if incorrect_outcomes:
            info["incorrect_outcomes"] = incorrect_outcomes

        # Return the result and any additional info
        return factual, info

    def is_necessary(
        self,
        env: StructuralCausalModel,
        event: dict,
        outcome: dict,
        state: dict,
        noise=None,
        **kwargs,
    ):
        """
        Check if the event is a necessary condition for the outcome obtained in the given state
        :param env: StructuralCausalModel
        :param event: dictionary of values of a given set of variables
        :param outcome: dictionary of values of the outcome variables
        :param state: dictionary of values of all observable variables
        :param noise: dictionary of values of all exogenous noise variables
        :param kwargs: additional arguments needed for the particular definition
        :return: answer: bool indicating whether the event is a necessary actual cause
        :return: info: dict with additional info about the necessity test
        """
        raise NotImplementedError("Necessity condition not implemented")

    def is_sufficient(
        self,
        env: StructuralCausalModel,
        event: dict,
        outcome: dict,
        state: dict,
        noise=None,
        **kwargs,
    ):
        """
        Check if the event is a sufficient condition for the outcome obtained in the given state
        :param env: StructuralCausalModel
        :param event: dictionary of values of a given set of variables
        :param outcome: dictionary of values of the outcome variables
        :param state: dictionary of values of all observable variables
        :param noise: dictionary of values of all exogenous noise variables
        :param kwargs: additional arguments needed for the particular definition
        :return: answer: bool indicating whether the event is a sufficient actual cause
        :return: info: dict with additional info about the sufficiency test
        """
        raise NotImplementedError("Sufficiency condition not implemented")

    def is_minimal(
        self,
        env: StructuralCausalModel,
        event: dict,
        outcome: dict,
        state: dict,
        noise=None,
        **kwargs,
    ):
        """
        Check if a given actual cause is minimal
        This is AC3 in the definition of actual cause
        :param env: StructuralCausalModel
        :param event: dictionary of values of a given set of variables
        :param outcome: dictionary of values of the outcome variables
        :param state: dictionary of values of all observable variables
        :param noise: dictionary of values of all exogenous noise variables
        :param kwargs: additional arguments needed for the particular definition
        :return: answer: bool indicating whether the event is a minimal actual cause
        :return: info: dict with any additional info about the minimality test
        """

        info = {}
        # Base case: singleton event
        if len(event.keys()) == 1:
            return True, info

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
                info = {"ac3_smaller_cause": subevent}
                return False, info

        return True, info

    def is_actual_cause(self, env, event, outcome, state, noise=None, **kwargs):
        """
        Check if the event is an actual cause of the outcome in the state
        Conditions are ordered by increasing computational cost
        :param env: StructuralCausalModel
        :param event: dictionary of values of a given set of variables
        :param outcome: dictionary of values of the outcome variables
        :param state: dictionary of values of all observable variables
        :param noise: dictionary of values of all exogenous noise variables
        :return: answer: bool indicating whether the event is an actual cause
        :return: info: dict with additional info about the actual causality test
        """

        # TODO - incorporate the witness set

        info = {
            "is_factual": False,
            "is_sufficient": None,
            "is_necessary": None,
            "is_minimal": None,
        }
        # Check for AC1
        ac1, ac1_info = self.is_factual(env, event, outcome, state, noise, **kwargs)
        add_info(info, ac1_info)

        # Stop if given event and outcome are not factual
        if not ac1:
            info["is_factual"] = False
            return False, info
        info["is_factual"] = True

        # Check for AC2b
        ac2b, ac2b_info = self.is_sufficient(
            env, event, outcome, state, noise, **kwargs
        )
        add_info(info, ac2b_info)
        if ac2b:
            info["is_sufficient"] = True
        else:
            info["is_sufficient"] = False

        # Check for AC2a
        ac2a, ac2a_info = self.is_necessary(env, event, outcome, state, noise, **kwargs)
        add_info(info, ac2a_info)
        if ac2a:
            info["is_necessary"] = True
        else:
            info["is_necessary"] = False

        # Stop if not necessary or sufficient
        if not ac2a or not ac2b:
            return False, info

        # Check for AC3
        ac3, ac3_info = self.is_minimal(env, event, outcome, state, noise, **kwargs)
        add_info(info, ac3_info)
        if ac3:
            info["is_minimal"] = True
            return True, info
        else:
            info["is_minimal"] = False
            return False, info

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
