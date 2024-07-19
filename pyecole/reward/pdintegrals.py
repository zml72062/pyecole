import ecole.reward
from ..scip import Model
from .base import BaseRewardFunction
from typing import *

class PrimalIntegral(BaseRewardFunction):
    """
    Primal integral difference.
    
    The reward is defined as the primal integral since the previous state, where 
    the integral is computed with respect to the solving time. The solving time 
    is specific to the operating system: it includes time spent in `reset()` and 
    time spent waiting on the agent.
    """
    def __init__(self, wall: bool = False, 
                 bound_function: Callable[[Model], Tuple[float, float]] = None
                 ) -> None:
        """
        Create a `PrimalIntegral` reward function.
        
        Parameters
        ----------
        wall:
            If true, the wall time will be used. If false (default), the process 
            time will be used.
        bound_function:
            A function which takes a `Model` and returns a tuple of an initial 
            primal bound and the offset to compute the primal bound with respect 
            to. Values should be ordered as `(offset, initial_primal_bound)`. 
            The default function returns `(0, -1e20)` if the problem is a 
            maximization and `(0, 1e20)` otherwise.
        """
        func = lambda model: bound_function(model.model)
        super().__init__(ecole.reward.PrimalIntegral(wall, func))

    def before_reset(self, model: Model) -> None:
        """
        Reset the internal clock counter and the event handler.
        """
        self.data.before_reset(model.model)

    def extract(self, model: Model, done: bool = False) -> float:
        """
        Computes the current primal integral and returns the difference.

        The difference is computed based on the dual integral between sequential 
        calls.
        """
        return self.data.extract(model.model, done)

class DualIntegral(BaseRewardFunction):
    """
    Dual integral difference.
    
    The reward is defined as the dual integral since the previous state, where 
    the integral is computed with respect to the solving time. The solving time 
    is specific to the operating system: it includes time spent in `reset()` and 
    time spent waiting on the agent.
    """
    def __init__(self, wall: bool = False, 
                 bound_function: Callable[[Model], Tuple[float, float]] = None
                 ) -> None:
        """
        Create a `DualIntegral` reward function.

        Parameters
        ----------
        wall:
            If true, the wall time will be used. If false (default), the process 
            time will be used.
        bound_function:
            A function which takes a `Model` and returns a tuple of an initial 
            dual bound and the offset to compute the dual bound with respect to. 
            Values should be ordered as `(offset, initial_dual_bound)`. The 
            default function returns `(0, 1e20)` if the problem is a maximization 
            and `(0, -1e20)` otherwise.
        """
        func = lambda model: bound_function(model.model)
        super().__init__(ecole.reward.DualIntegral(wall, func))

    def before_reset(self, model: Model) -> None:
        """
        Reset the internal clock counter and the event handler.
        """
        self.data.before_reset(model.model)

    def extract(self, model: Model, done: bool = False) -> float:
        """
        Computes the current dual integral and returns the difference.

        The difference is computed based on the dual integral between sequential calls.
        """
        return self.data.extract(model.model, done)

class PrimalDualIntegral(BaseRewardFunction):
    """
    Primal-dual integral difference.

    The reward is defined as the primal-dual integral since the previous state, 
    where the integral is computed with respect to the solving time. The solving 
    time is specific to the operating system: it includes time spent in `reset()` 
    and time spent waiting on the agent.    
    """
    def __init__(self, wall: bool = False, 
                 bound_function: Callable[[Model], Tuple[float, float]] = None
                 ) -> None:
        """
        Create a `PrimalDualIntegral` reward function.

        Parameters
        ----------
        wall:
            If true, the wall time will be used. If false (default), the process 
            time will be used.
        bound_function:
            A function which takes a `Model` and returns a tuple of an initial 
            primal bound and dual bound. Values should be ordered as 
            `(initial_primal_bound, initial_dual_bound)`. The default function 
            returns `(-1e20, 1e20)` if the problem is a maximization and 
            `(1e20, -1e20)` otherwise.
        """
        func = lambda model: bound_function(model.model)
        super().__init__(ecole.reward.PrimalDualIntegral(wall, func))

    def before_reset(self, model: Model) -> None:
        """
        Reset the internal clock counter and the event handler.
        """
        self.data.before_reset(model.model)

    def extract(self, model: Model, done: bool = False) -> float:
        """
        Computes the current primal-dual integral and returns the difference.

        The difference is computed based on the primal-dual integral between sequential calls.
        """
        return self.data.extract(model.model, done)
    
    