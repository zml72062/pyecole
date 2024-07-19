import ecole.observation
import numpy as np
from ..scip import Model
from ..typing import ObservationFunction
from typing import Optional


class Pseudocosts(ObservationFunction):
    """
    Pseudocosts observation function on branch-and-bound nodes.
    
    This observation obtains pseudocosts for all LP fractional candidate variables 
    at a branch-and-bound node. The pseudocost is a cheap approximation to the 
    strong branching score and measures the quality of branching for each 
    variable. This observation can be used as a practical branching strategy by 
    always branching on the variable with the highest pseudocost, although in 
    practice is it not as efficient as SCIP's default strategy, reliability 
    pseudocost branching (also known as hybrid branching).

    This observation function extracts an array containing the pseudocost for each 
    variable in the problem. Variables are ordered according to their position in
    the original problem (`SCIPvarGetProbindex`), hence they can be indexed by 
    the `Branching` environment `action_set`. Variables for which a pseudocost 
    is not applicable are filled with `NaN`.
    """
    def __init__(self) -> None:
        self.func = ecole.observation.Pseudocosts()

    def before_reset(self, model: Model) -> None:
        """
        Do nothing.
        """
        self.func.before_reset(model.model)

    def extract(self, model: Model, done: bool) -> Optional[np.ndarray]:
        """        
        Extract an array containing pseudocosts.
        """
        return self.func.extract(model.model, done)

