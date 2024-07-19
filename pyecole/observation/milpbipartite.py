import ecole.observation
import numpy as np
from ..scip import Model
from ..typing import ObservationFunction
from .coo_matrix import coo_matrix
from typing import Optional
from enum import Enum

class MilpBipartiteObs:
    """
    Bipartite graph observation that represents the most recent MILP during 
    presolving.

    The optimization problem is represented as an heterogenous bipartite graph. 
    On one side, a node is associated with one variable, on the other side a 
    node is associated with one constraint. There exists an edge between a 
    variable and a constraint if the variable exists in the constraint with a 
    non-zero coefficient.

    Each variable and constraint node is associated with a vector of features. 
    Each edge is associated with the coefficient of the variable in the 
    constraint.
    """
    def __init__(self, data: ecole.observation.MilpBipartiteObs) -> None:
        self.data = data

    @property
    def edge_features(self) -> coo_matrix:
        """
        The constraint matrix of the optimization problem, with rows for 
        contraints and columns for variables.
        """
        return self.data.edge_features
    
    class ConstraintFeatures(Enum):
        bias = 0

    @property
    def constraint_features(self) -> np.ndarray:
        """
        A matrix where each row represents a constraint, and each column a feature 
        of the constraints.
        """
        return self.data.constraint_features
    
    class VariableFeatures(Enum):
        objective = 0
        is_type_binary = 1
        is_type_integer = 2
        is_type_implicit_integer = 3
        is_type_continuous = 4
        has_lower_bound = 5
        has_upper_bound = 6
        lower_bound = 7
        upper_bound = 8

    @property
    def variable_features(self) -> np.ndarray:
        """
        A matrix where each row represents a variable, and each column a feature 
        of the variable.

        Variables are ordered according to their position in the original problem 
        (`SCIPvarGetProbindex`), hence they can be indexed by the `Branching` 
        environment `action_set`.
        """
        return self.data.variable_features


class MilpBipartite(ObservationFunction):
    """
    Bipartite graph observation function for the sub-MILP at the latest 
    branch-and-bound node.

    This observation function extracts structured `MilpBipartiteObs`.
    """
    def __init__(self, normalize: bool = False) -> None:
        """
        Constructor for `MilpBipartite`.

        Parameters
        ----------
        normalize:
            Should the features be normalized? This is recommended for some 
            applications such as deep learning models.
        """
        self.func = ecole.observation.MilpBipartite(normalize)

    def before_reset(self, model: Model) -> None:
        """
        Do nothing.
        """
        self.func.before_reset(model.model)

    def extract(self, model: Model, done: bool) -> Optional[MilpBipartiteObs]:
        """
        Extract a new `MilpBipartiteObs`.        
        """
        data = self.func.extract(model.model, done)
        if data is not None:
            return MilpBipartiteObs(data)
        return data
    
