import ecole.observation
import numpy as np
from ..scip import Model
from ..typing import ObservationFunction
from .coo_matrix import coo_matrix
from typing import Optional
from enum import Enum

class NodeBipartiteObs:
    """
    Bipartite graph observation for branch-and-bound nodes.

    The optimization problem is represented as an heterogenous bipartite graph. 
    On one side, a node is associated with one variable, on the other side a 
    node is associated with one LP row. There exist an edge between a variable 
    and a constraint if the variable exists in the constraint with a non-zero 
    coefficient.

    Each variable and constraint node is associated with a vector of features. 
    Each edge is associated with the coefficient of the variable in the 
    constraint.
    """
    def __init__(self, data: ecole.observation.NodeBipartiteObs) -> None:
        self.data = data

    @property
    def edge_features(self) -> coo_matrix:
        """
        The constraint matrix of the optimization problem, with rows for 
        contraints and columns for variables.
        """
        return self.data.edge_features

    class RowFeatures(Enum):
        bias = 0
        objective_cosine_similarity = 1
        is_tight = 2
        dual_solution_value = 3
        scaled_age = 4

    @property
    def row_features(self) -> np.ndarray:
        """
        A matrix where each row represents a constraint, and each column a 
        feature of the constraints.
        """
        return self.data.row_features
    
    class ColumnFeatures(Enum):
        objective = 0
        is_type_binary = 1
        is_type_integer = 2
        is_type_implicit_integer = 3
        is_type_continuous = 4
        has_lower_bound = 5
        has_upper_bound = 6
        normed_reduced_cost = 7
        solution_value = 8
        solution_frac = 9
        is_solution_at_lower_bound = 10
        is_solution_at_upper_bound = 11
        scaled_age = 12
        incumbent_value = 13
        average_incumbent_value = 14
        is_basis_lower = 15
        is_basis_basic = 16
        is_basis_upper = 17
        is_basis_zero = 18

    @property
    def column_features(self) -> np.ndarray:
        """
        A matrix where each row represents a variable, and each column a 
        feature of the variable.

        Variables are ordered according to their position in the original 
        problem (`SCIPvarGetProbindex`), hence they can be indexed by the 
        `Branching` environment `action_set`.
        """
        return self.data.column_features
    

class NodeBipartite(ObservationFunction):
    """
    Bipartite graph observation function on branch-and-bound node.

    This observation function extracts structured `NodeBipartiteObs`.
    """
    def __init__(self, cache: bool = False) -> None:
        """
        Constructor for `NodeBipartite`.

        Parameters
        ----------
        cache:
            Whether or not to cache static features within an episode. 
            Currently, this is only safe if cutting planes are disabled.
        """
        self.func = ecole.observation.NodeBipartite(cache)
    
    def before_reset(self, model: Model) -> None:
        """
        Cache some feature not expected to change during an episode.
        """
        self.func.before_reset(model.model)

    def extract(self, model: Model, done: bool) -> Optional[NodeBipartiteObs]:
        """
        Extract a new `NodeBipartiteObs`.
        """
        data = self.func.extract(model.model, done)
        if data is not None:
            return NodeBipartiteObs(data)
        return data

