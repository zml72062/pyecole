import ecole.dynamics
from ..typing import Dynamics
from ..scip.model import Model
from ..random import RandomEngine
from typing import *
import numpy as np

class BranchingDynamics(Dynamics):
    """
    Single variable branching Dynamics.

    Based on a SCIP `branching callback <https://www.scipopt.org/doc/html/BRANCH.php>`_
    with maximal priority and no depth limit.
    The dynamics give the control back to the user every time the callback would be called.
    The user receives as an action set the list of branching candidates, and is expected to select
    one of them as the action.
    """
    def __init__(self, pseudo_candidates: bool = False) -> None:
        """
        Create new dynamics.

        Parameters
        ----------
        pseudo_candidates:
            Whether the action set contains pseudo branching variable candidates (``SCIPgetPseudoBranchCands``)
            or LP branching variable candidates (``SCIPgetPseudoBranchCands``).
        """
        self.dyn = ecole.dynamics.BranchingDynamics(pseudo_candidates)

    def reset_dynamics(self, model: Model) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Start solving up to first branching node.

        Start solving with SCIP defaults (``SCIPsolve``) and give back control to the user on the
        first branching decision.
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
                This can happen without branching, for instance if the instance is solved during presolving.
            action_set:
                List of indices of branching candidate variables.
                Available candidates depend on parameters in :py:meth:`__init__`.
                Variable indices (values in the ``action_set``) are their position in the original problem
                (``SCIPvarGetProbindex``).
                Variable ordering in the ``action_set`` is arbitrary.
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
    
    def step_dynamics(self, model: Model, action: int
                      ) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Branch and resume solving until next branching.

        Branching is done on a single variable using ``SCIPbranchVar``.
        The control is given back to the user on the next branching decision or when done.

        Parameters
        ----------
            model:
                The state of the Markov Decision Process. Passed by the environment.
            action:
                The index the LP column of the variable to branch on. One element of the action set.
                If an explicit ``ecole.Default`` is passed, then default SCIP branching is used, that is, the next
                branching rule is used fetch by SCIP according to their priorities.

        Returns
        -------
            done:
                Whether the instance is solved.
            action_set:
                List of indices of branching candidate variables.
                Available candidates depend on parameters in :py:meth:`__init__`.
                Variable indices (values in the ``action_set``) are their position in the original problem
                (``SCIPvarGetProbindex``).
                Variables ordering in the ``action_set`` is arbitrary.
        """
        return self.dyn.step_dynamics(model.model, action)
    
    