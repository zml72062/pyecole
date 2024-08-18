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
        """
        Enum class indicating meanings of constraint features.
        """
        bias = 0
        """Bias term of the constraint."""
        objective_cosine_similarity = 1
        """Cosine similarity between the constraint and the LP objective."""
        is_tight = 2
        """Whether the constraint is tight with current LP solution."""
        dual_solution_value = 3
        """Dual solution value corresponding to the constraint at current LP."""
        scaled_age = 4
        """Age of the constraint, i.e. for how many iterations has the constraint
        stayed non-tight during the process of solving current LP."""

    @property
    def row_features(self) -> np.ndarray:
        """
        A matrix where each row represents a constraint, and each column a 
        feature of the constraints.
        """
        return self.data.row_features
    
    class ColumnFeatures(Enum):
        """
        Enum class indicating meanings of variable features.
        """
        objective = 0
        """Objective coefficient of the variable."""
        is_type_binary = 1
        """Whether the variable is binary."""
        is_type_integer = 2
        """Whether the variable is integral."""
        is_type_implicit_integer = 3
        """Whether the variable is implicitly integral."""
        is_type_continuous = 4
        """Whether the variable is continuous."""
        has_lower_bound = 5
        """Whether the variable has a finite lower bound."""
        has_upper_bound = 6
        """Whether the variable has a finite upper bound."""
        normed_reduced_cost = 7
        """Reduced cost of the variable at current LP."""
        solution_value = 8
        """Primal solution value for the variable at current LP."""
        solution_frac = 9
        """Fractional part of primal solution value for the variable at current
        LP, if the variable is restricted to be integral. For continuous variables
        this entry would be 0."""
        is_solution_at_lower_bound = 10
        """Whether current LP solution of the variable is at its lower bound."""
        is_solution_at_upper_bound = 11
        """Whether current LP solution of the variable is at its upper bound."""
        scaled_age = 12
        """Age of the variable, i.e. for how many iterations has the variable
        stayed 0.0 during the process of solving current LP."""
        incumbent_value = 13
        """Incumbent (best so far) solution to the MILP for the variable. If there
        is currently no incumbent solution, this entry would be NaN."""
        average_incumbent_value = 14
        """Average incumbent solution to the MILP for the variable. If there is
        currently no incumbent solution, this entry would be NaN."""
        is_basis_lower = 15
        """Whether the variable is a basis variable taking value at its lower bound 
        for current LP."""
        is_basis_basic = 16
        """Whether the variable is a basis variable taking value neither at its 
        lower bound nor upper bound for current LP."""
        is_basis_upper = 17
        """Whether the variable is a basis variable taking value at its upper bound 
        for current LP."""
        is_basis_zero = 18
        """Whether the variable is a non-basis variable (thus taking value 0.0) for
        current LP."""
        lower_bound = 19
        """Lower bound of the variable (if finite). If a lower bound does not exist, 
        this entry would be 0."""
        upper_bound = 20
        """Upper bound of the variable (if finite). If an upper bound does not exist,
        this entry would be 0."""

    @property
    def column_features(self) -> np.ndarray:
        """
        A matrix where each row represents a variable, and each column a 
        feature of the variable.

        Variables are ordered according to their position in the original 
        problem (`SCIPvarGetProbindex`), hence they can be indexed by the 
        `Branching` environment `action_set`.
        """
        return self.data.variable_features
    

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

