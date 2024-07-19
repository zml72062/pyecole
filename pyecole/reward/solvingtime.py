import ecole.reward
from ..scip import Model
from .base import BaseRewardFunction

class SolvingTime(BaseRewardFunction):
    """
    Solving time difference.
    
    The reward is defined as the number of seconds spent solving the instance 
    since the previous state. The solving time is specific to the operating 
    system: it includes time spent in `reset()` and time spent waiting on the 
    agent.
    """
    def __init__(self, wall: bool = False) -> None:
        """
        Create a `SolvingTime` reward function.

        Parameters
        ----------
        wall: 
            If true, the wall time will be used. If false (default), the process 
            time will be used.
        """
        super().__init__(ecole.reward.SolvingTime(wall))

    def before_reset(self, model: Model) -> None:
        """
        Reset the internal clock counter.
        """
        self.data.before_reset(model.model)

    def extract(self, model: Model, done: bool = False) -> float:
        """
        Update the internal clock counter and return the difference.
        
        The difference in solving time is computed in between calls.
        """
        return self.data.extract(model.model, done)

