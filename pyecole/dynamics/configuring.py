import ecole.dynamics
from ..typing import Dynamics
from ..scip.model import Model
from ..random import RandomEngine
from typing import *

class ConfiguringDynamics(Dynamics):
    """
    Setting solving parameters Dynamics.

    These dynamics are meant to be used as a (contextual) bandit to find good parameters for SCIP.
    """
    def __init__(self) -> None:
        self.dyn = ecole.dynamics.ConfiguringDynamics()

    def reset_dynamics(self, model: Model) -> Tuple[bool, None]:
        """
        Does nothing.

        Users can inherit from this dynamics to change when in the solving process parameters will be set
        (for instance after presolving).

        Parameters
        ----------
            model:
                The state of the Markov Decision Process. Passed by the environment.

        Returns
        -------
            done:
                Whether the instance is solved. Always false.
            action_set:
                Unused.
        """
        return self.dyn.reset_dynamics(model.model)

    def set_dynamics_random_state(self, model: Model, rng: RandomEngine) -> None:
        """
        Set seeds on the :py:class:`Model`.

        Set seed parameters, including permutation, LP, and shift.

        Parameters
        ----------
            model:
                The state of the Markov Decision Process. Passed by the environment.
            rng:
                The source of randomness. Passed by the environment.
        """
        self.dyn.set_dynamics_random_state(model.model, rng)

    def step_dynamics(self, model: Model, 
                      action: Dict[str, Union[bool, int, float, str]]
                      ) -> Tuple[bool, None]:
        """
        Set parameters and solve the instance.

        Parameters
        ----------
            model:
                The state of the Markov Decision Process. Passed by the environment.
            action:
                A mapping of parameter names and values.

        Returns
        -------
            done:
                Whether the instance is solved. Always true.
            action_set:
                Unused.
        """
        return self.dyn.step_dynamics(model.model, action)
        
