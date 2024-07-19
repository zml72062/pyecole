import ecole.observation
import numpy as np
from ..scip import Model
from ..typing import ObservationFunction
from typing import Optional


class StrongBranchingScores(ObservationFunction):
    """
    Strong branching score observation function on branch-and-bound node.
    
    This observation obtains scores for all LP or pseudo candidate variables at 
    a branch-and-bound node. The strong branching score measures the quality of 
    each variable for branching (higher is better). This observation can be used 
    as an expert for imitation learning algorithms.

    This observation function extracts an array containing the strong branching 
    score for each variable in the problem. Variables are ordered according to 
    their position in the original problem (`SCIPvarGetProbindex`), hence they 
    can be indexed by the `Branching` environment `action_set`. Variables for 
    which a strong branching score is not applicable are filled with `NaN`.
    """
    def __init__(self, pseudo_candidates: bool = False) -> None:
        """
        Constructor for `StrongBranchingScores`.
        
        Parameters
        ----------
        pseudo_candidates:
            The parameter determines if strong branching scores are computed for 
            pseudo candidate variables (when true) or LP candidate variables 
            (when false).
        """
        self.func = ecole.observation.StrongBranchingScores(pseudo_candidates)
    
    def before_reset(self, model: Model) -> None:
        """
        Do nothing.
        """
        self.func.before_reset(model.model)

    def extract(self, model: Model, done: bool) -> Optional[np.ndarray]:
        """
        Extract an array containing strong branching scores.
        """
        return self.func.extract(model.model, done)
        
