import ecole.observation
import numpy as np
from ..scip import Model
from ..typing import ObservationFunction
from typing import Optional
from enum import Enum

class Khalil2016Obs:
    """
    Branching candidates features from Khalil et al. (2016).

    The observation is a matrix where rows represent all variables and columns represent features related
    to these variables.
    See [Khalil2016]_ for a complete reference on this observation function.

    .. [Khalil2016]
        Khalil, Elias Boutros, Pierre Le Bodic, Le Song, George Nemhauser, and Bistra Dilkina.
        "`Learning to branch in mixed integer programming.
        <https://dl.acm.org/doi/10.5555/3015812.3015920>`_"
        *Thirtieth AAAI Conference on Artificial Intelligence*. 2016.
    """
    def __init__(self, data: ecole.observation.Khalil2016Obs) -> None:
        self.data = data

    class Features(Enum):
        obj_coef = 0
        obj_coef_pos_part = 1
        obj_coef_neg_part = 2
        n_rows = 3
        rows_deg_mean = 4
        rows_deg_stddev = 5
        rows_deg_min = 6
        rows_deg_max = 7
        rows_pos_coefs_count = 8
        rows_pos_coefs_mean = 9
        rows_pos_coefs_stddev = 10
        rows_pos_coefs_min = 11
        rows_pos_coefs_max = 12
        rows_neg_coefs_count = 13
        rows_neg_coefs_mean = 14
        rows_neg_coefs_stddev = 15
        rows_neg_coefs_min = 16
        rows_neg_coefs_max = 17
        slack = 18
        ceil_dist = 19
        pseudocost_up = 20
        pseudocost_down = 21
        pseudocost_ratio = 22
        pseudocost_sum = 23
        pseudocost_product = 24
        n_cutoff_up = 25
        n_cutoff_down = 26
        n_cutoff_up_ratio = 27
        n_cutoff_down_ratio = 28
        rows_dynamic_deg_mean = 29
        rows_dynamic_deg_stddev = 30
        rows_dynamic_deg_min = 31
        rows_dynamic_deg_max = 32
        rows_dynamic_deg_mean_ratio = 33
        rows_dynamic_deg_min_ratio = 34
        rows_dynamic_deg_max_ratio = 35
        coef_pos_rhs_ratio_min = 36
        coef_pos_rhs_ratio_max = 37
        coef_neg_rhs_ratio_min = 38
        coef_neg_rhs_ratio_max = 39
        pos_coef_pos_coef_ratio_min = 40
        pos_coef_pos_coef_ratio_max = 41
        pos_coef_neg_coef_ratio_min = 42
        pos_coef_neg_coef_ratio_max = 43
        neg_coef_pos_coef_ratio_min = 44
        neg_coef_pos_coef_ratio_max = 45
        neg_coef_neg_coef_ratio_min = 46
        neg_coef_neg_coef_ratio_max = 47
        active_coef_weight1_count = 48
        active_coef_weight1_sum = 49
        active_coef_weight1_mean = 50
        active_coef_weight1_stddev = 51
        active_coef_weight1_min = 52
        active_coef_weight1_max = 53
        active_coef_weight2_count = 54
        active_coef_weight2_sum = 55
        active_coef_weight2_mean = 56
        active_coef_weight2_stddev = 57
        active_coef_weight2_min = 58
        active_coef_weight2_max = 59
        active_coef_weight3_count = 60
        active_coef_weight3_sum = 61
        active_coef_weight3_mean = 62
        active_coef_weight3_stddev = 63
        active_coef_weight3_min = 64
        active_coef_weight3_max = 65
        active_coef_weight4_count = 66
        active_coef_weight4_sum = 67
        active_coef_weight4_mean = 68
        active_coef_weight4_stddev = 69
        active_coef_weight4_min = 70
        active_coef_weight4_max = 71

    @property
    def features(self) -> np.ndarray:
        """
        A matrix where each row represents a variable, and each column a feature 
        of the variable.
        
        Variables are ordered according to their position in the original problem 
        (`SCIPvarGetProbindex`), hence they can be indexed by the `Branching` 
        environment `action_set`. Variables for which the features are not 
        applicable are filled with `NaN`.


        The first `Khalil2016Obs.n_static_features` features columns are static 
        (they do not change through the solving process), and the remaining 
        `Khalil2016Obs.n_dynamic_features` are dynamic.
        """
        return self.data.features

    @property
    def n_static_features(self) -> int:
        return self.data.n_static_features

    @property
    def n_dynamic_features(self) -> int:
        return self.data.n_dynamic_features


class Khalil2016(ObservationFunction):
    """
    Branching candidates features from Khalil et al. (2016).
    
    This observation function extract structured `Khalil2016Obs`.
    """
    def __init__(self, pseudo_candidates: bool = False) -> None:
        """
        Create new observation.
        
        Parameters
        ----------
        pseudo_candidates:
            Whether the pseudo branching variable candidates or LP branching 
            variable candidates are observed.
        """
        self.func = ecole.observation.Khalil2016(pseudo_candidates)
    
    def before_reset(self, model: Model) -> None:
        """
        Reset static features cache.
        """
        self.func.before_reset(model.model)
    
    def extract(self, model: Model, done: bool) -> Optional[Khalil2016Obs]:
        """
        Extract the observation matrix.
        """
        data = self.func.extract(model.model, done)
        if data is not None:
            return Khalil2016Obs(data)
        return data
    
