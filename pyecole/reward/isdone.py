import ecole.reward
from ..scip import Model
from .base import BaseRewardFunction

class IsDone(BaseRewardFunction):
    """
    Single reward on terminal states.
    """
    def __init__(self) -> None:
        super().__init__(ecole.reward.IsDone())
    
    def before_reset(self, model: Model) -> None:
        """
        Do nothing.
        """
        self.data.before_reset(model.model)

    def extract(self, model: Model, done: bool = False) -> float:
        """
        Return 1 if the episode is on a terminal state, 0 otherwise.
        """
        return self.data.extract(model.model, done)
    
