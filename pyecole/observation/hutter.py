import ecole.observation
import numpy as np
from ..scip import Model
from ..typing import ObservationFunction
from typing import Optional
from enum import Enum

class Hutter2011Obs:
    """
    Instance features from Hutter et al. (2011).

    The observation is a vector of features that globally characterize the instance.
    See [Hutter2011]_ for a complete reference on this observation function.

    .. [Hutter2011]
        Hutter, Frank, Hoos, Holger H., and Leyton-Brown, Kevin.
        "`Sequential model-based optimization for general algorithm configuration.
        <https://doi.org/10.1007/978-3-642-25566-3_40>`_"
        *International Conference on Learning and Intelligent Optimization*. 2011.
    """
    def __init__(self, data: ecole.observation.Hutter2011Obs) -> None:
        self.data = data
    
    class Features(Enum):
        nb_variables = 0
        nb_constraints = 1
        nb_nonzero_coefs = 2
        variable_node_degree_mean = 3
        variable_node_degree_max = 4
        variable_node_degree_min = 5
        variable_node_degree_std = 6
        constraint_node_degree_mean = 7
        constraint_node_degree_max = 8
        constraint_node_degree_min = 9
        constraint_node_degree_std = 10
        node_degree_mean = 11
        node_degree_max = 12
        node_degree_min = 13
        node_degree_std = 14
        node_degree_25q = 15
        node_degree_75q = 16
        edge_density = 17
        lp_slack_mean = 18
        lp_slack_max = 19
        lp_slack_l2 = 20
        lp_objective_value = 21
        objective_coef_m_std = 22
        objective_coef_n_std = 23
        objective_coef_sqrtn_std = 24
        constraint_coef_mean = 25
        constraint_coef_std = 26
        constraint_var_coef_mean = 27
        constraint_var_coef_std = 28
        discrete_vars_support_size_mean = 29
        discrete_vars_support_size_std = 30
        ratio_unbounded_discrete_vars = 31
        ratio_continuous_vars = 32

    @property
    def features(self) -> np.ndarray:
        """
        A vector of instance features.
        """
        return self.data.features

class Hutter2011(ObservationFunction):
    """
    Instance features from Hutter et al. (2011).

    This observation function extracts a structured `Hutter2011Obs`.
    """
    def __init__(self) -> None:
        self.func = ecole.observation.Hutter2011()

    def before_reset(self, model: Model) -> None:
        """
        Do nothing.
        """
        self.func.before_reset(model.model)
    
    def extract(self, model: Model, done: bool) -> Optional[Hutter2011Obs]:
        """
        Extract the observation matrix.
        """
        data = self.func.extract(model.model, done)
        if data is not None:
            return Hutter2011Obs(data)
        return data

    