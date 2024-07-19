import ecole.reward
from ..scip import Model
from .base import BaseRewardFunction

class LpIterations(BaseRewardFunction):
    """
    LP iterations difference.

    The reward is defined as the number of iterations spent in solving the 
    Linear Programs associated with the problem since the previous state.
    """
    def __init__(self) -> None:
        super().__init__(ecole.reward.LpIterations())

    def before_reset(self, model: Model) -> None:
        """
        Reset the internal LP iterations count.
        """
        self.data.before_reset(model.model)

    def extract(self, model: Model, done: bool = False) -> float:
        """
        Update the internal LP iteration count and return the difference.
        
        The difference in LP iterations is computed in between calls.
        """
        return self.data.extract(model.model, done)

