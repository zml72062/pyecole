import ecole.reward
from ..scip import Model
from .base import BaseRewardFunction

class NNodes(BaseRewardFunction):
    """
    Number of nodes difference.
    
    The reward is defined as the total number of nodes processed since the 
    previous state.
    """
    def __init__(self) -> None:
        super().__init__(ecole.reward.NNodes())

    def before_reset(self, model: Model) -> None:
        """
        Reset the internal node count.
        """
        self.data.before_reset(model.model)

    def extract(self, model: Model, done: bool = False) -> float:
        """
        Update the internal node count and return the difference.

        The difference in number of nodes is computed in between calls.
        """
        return self.data.extract(model.model, done)

