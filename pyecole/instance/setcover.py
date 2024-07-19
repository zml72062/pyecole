import ecole.instance
from ..typing import InstanceGenerator
from ..random import RandomEngine
from ..scip.model import Model

class SetCoverGenerator(InstanceGenerator):
    def __init__(self, n_rows: int = 500, 
                 n_cols: int = 1000, 
                 density: float = 0.05, 
                 max_coef: int = 100, 
                 rng: RandomEngine = None) -> None:
        """
        Generate a set cover MILP problem instance.

        Algorithm described in [Balas1980]_.

        Parameters
        ----------
        n_rows:
            The number of rows.
        n_cols:
            The number of columns.
        density:
            The density of the constraint matrix.
            The value must be in the range ]0,1].
        max_coef:
            Maximum objective coefficient.
            The value must be greater than one.
        rng:
            The random number generator used to perform all sampling.

        References
        ----------
            .. [Balas1980]
                Egon Balas and Andrew Ho.
                "Set covering algorithms using cutting planes, heuristics, and subgradient optimization: A computational study".
                *Mathematical Programming*, 12, pp. 37-60. 1980.
        """
        self.generator = ecole.instance.SetCoverGenerator(
            n_rows, n_cols, density, max_coef, 
            rng.generator if rng is not None else rng
        )
    
    @property
    def n_rows(self) -> int:
        return self.generator.n_rows

    @property
    def n_cols(self) -> int:
        return self.generator.n_cols

    @property
    def density(self) -> float:
        return self.generator.density

    @property
    def max_coef(self) -> int:
        return self.generator.max_coef

    @staticmethod
    def generate_instance(n_rows: int = 500, 
                          n_cols: int = 1000, 
                          density: float = 0.05, 
                          max_coef: int = 100, 
                          *, rng: RandomEngine) -> Model:
        """
        Generate a set cover MILP problem instance.

        Algorithm described in [Balas1980]_.

        Parameters
        ----------
        n_rows:
            The number of rows.
        n_cols:
            The number of columns.
        density:
            The density of the constraint matrix.
            The value must be in the range ]0,1].
        max_coef:
            Maximum objective coefficient.
            The value must be greater than one.
        rng:
            The random number generator used to perform all sampling.

        References
        ----------
            .. [Balas1980]
                Egon Balas and Andrew Ho.
                "Set covering algorithms using cutting planes, heuristics, and subgradient optimization: A computational study".
                *Mathematical Programming*, 12, pp. 37-60. 1980.
        """
        return Model(
            ecole.instance
                 .SetCoverGenerator
                 .generate_instance(n_rows, n_cols, density, 
                                    max_coef, rng.generator
                                    if rng is not None else rng)
        )
        
    def seed(self, seed: int) -> None:
        self.generator.seed(seed)

    def __iter__(self) -> "SetCoverGenerator":
        return self
    
    def __next__(self) -> Model:
        return Model(next(self.generator))
    
