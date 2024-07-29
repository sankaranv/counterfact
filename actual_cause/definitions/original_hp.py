from actual_cause.definitions import ACDefinition
from actual_cause.causal_models.scm import StructuralCausalModel


class OriginalHP(ACDefinition):

    def __init__(self):
        super().__init__()

    def is_necessary(self, event, state, outcome, noise=None):
        """
        Check if the event is a necessary condition for the outcome in the state
        :param event: str
        :param state: Dict[str, Union[float, Callable[[], float]]]
        :param outcome: str
        :param noise: Dict[str, float]
        :return: bool
        """
        raise NotImplementedError

    def is_sufficient(self, event, state, outcome, noise=None):
        """
        Check if the event is a sufficient condition for the outcome in the state
        :param event: str
        :param state: Dict[str, Union[float, Callable[[], float]]]
        :param outcome: str
        :param noise: Dict[str, float]
        :return: bool
        """
        raise NotImplementedError
