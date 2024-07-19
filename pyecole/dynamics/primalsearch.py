import ecole.dynamics
from ..typing import Dynamics
from ..scip.model import Model
from ..random import RandomEngine
from typing import *
import numpy as np

class PrimalSearchDynamics(Dynamics):
    """
    Search for primal solutions Dynamics.

    Based on a SCIP `primal heuristic <https://www.scipopt.org/doc/html/HEUR.php>`_
    callback with maximal priority, which executes
    after the processing of a node is finished (``SCIP_HEURTIMING_AFTERNODE``).
    The dynamics give the control back to the user a few times (trials) each time
    the callback is called. The agent receives as an action set the list of all non-fixed
    discrete variables at the current node (pseudo branching candidates), and is
    expected to give back as an action a partial primal solution, i.e., a value
    assignment for a subset of these variables.
    """
    def __init__(self, trials_per_node: int = 1, 
                 depth_freq: int = 1, 
                 depth_start: int = 0, 
                 depth_stop: int = -1) -> None:
        """
        Initialize new `PrimalSearchDynamics`.

        Parameters
        ----------
            trials_per_node:
                Number of primal searches performed at each node (or -1 for an infinite number of trials).
            depth_freq:
                Depth frequency of when the primal search is called (``HEUR_FREQ`` in SCIP).
            depth_start:
                Tree depth at which the primal search starts being called (``HEUR_FREQOFS`` in SCIP).
            depth_stop:
                Tree depth after which the primal search stops being called (``HEUR_MAXDEPTH`` in SCIP).
        """
        self.dyn = ecole.dynamics.PrimalSearchDynamics(trials_per_node,
                                                       depth_freq,
                                                       depth_start,
                                                       depth_stop)
        
    def reset_dynamics(self, model: Model) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Start solving up to first primal heuristic call.

        Start solving with SCIP defaults (``SCIPsolve``) and give back control to the user on the
        first heuristic call.
        Users can inherit from this dynamics to change the defaults settings such as presolving
        and cutting planes.

        Parameters
        ----------
            model:
                The state of the Markov Decision Process. Passed by the environment.

        Returns
        -------
            done:
                Whether the instance is solved.
                This can happen before the heuristic gets called, for instance if the instance is solved during presolving.
            action_set:
                List of non-fixed discrete variables (``SCIPgetPseudoBranchCands``).
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
        self.dyn.set_dynamics_random_state(model.model, rng.generator)

    def step_dynamics(self, model: Model, action: np.ndarray
                      ) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Try to obtain a feasible primal solution from the given (partial) primal solution.

        If the number of search trials per node is exceeded, then continue solving until
        the next time the heuristic gets called.

        To obtain a complete feasible solution, variables are fixed to their partial assignment
        values, and the rest of the variable assigments is deduced by solving an LP in probing
        mode. If the provided partial assigment is empty, then nothing is done.

        Parameters
        ----------
            model:
                The state of the Markov Decision Process. Passed by the environment.
            action:
                A subset of the variables given in the action set, and their assigned values.

        Returns
        -------
            done:
                Whether the instance is solved.
            action_set:
                List of non-fixed discrete variables (``SCIPgetPseudoBranchCands``).
        """
        return self.dyn.step_dynamics(model.model, action)
        
