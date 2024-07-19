import ecole.instance
from ..typing import InstanceGenerator
from ..random import RandomEngine
from ..scip.model import Model
from typing import *

class CapacitatedFacilityLocationGenerator(InstanceGenerator):
    def __init__(self, n_customers: int = 100, 
                 n_facilities: int = 100, 
                 continuous_assignment: bool = True, 
                 ratio: float = 5.0, 
                 demand_interval: Tuple[int, int] = (5, 36), 
                 capacity_interval: Tuple[int, int] = (10, 161), 
                 fixed_cost_cste_interval: Tuple[int, int] = (0, 91), 
                 fixed_cost_scale_interval: Tuple[int, int] = (100, 111), 
                 rng: RandomEngine = None) -> None:
        """
        Generate a capacitated facility location MILP problem instance.

        The capacitated facility location assigns a number of customers to be served from a number of facilities.
        Not all facilities need to be opened.
        In fact, the problem is to minimized the sum of the fixed costs for each facilities and the sum of transportation
        costs for serving a given customer from a given facility.
        In a variant of the problem, the customers can be served from multiple facilities and the associated variables
        become [0,1] continuous.

        The sampling algorithm is described in [Cornuejols1991]_, but uniform sampling as been replaced by *integer*
        uniform sampling.

        Parameters
        ----------
        n_customers:
            The number of customers.
        n_facilities:
            The number of facilities.
        continuous_assignment:
            Whether variable for assigning a customer to a facility are binary or [0,1] continuous.
        ratio:
            After all sampling is performed, the capacities are scaled by `ratio * sum(demands) / sum(capacities)`.
        demand_interval:
            The customer demands are sampled independently as uniform integers in this interval [lower, upper[.
        capacity_interval:
            The facility capacities are sampled independently as uniform integers in this interval [lower, upper[.
        fixed_cost_cste_interval:
            The fixed costs are the sum of two terms.
            The first terms in the fixed costs for opening facilities are sampled independently as uniform integers
            in this interval [lower, upper[.
        fixed_cost_scale_interval:
            The fixed costs are the sum of two terms.
            The second terms in the fixed costs for opening facilities are sampled independently as uniform integers
            in this interval [lower, upper[ multiplied by the square root of their capacity prior to scaling.
            This second term reflects the economies of scale.
        rng:
            The random number generator used to peform all sampling.

        References
        ----------
        .. [Cornuejols1991]
            Cornuejols G, Sridharan R, Thizy J-M.
            "A Comparison of Heuristics and Relaxations for the Capacitated Plant Location Problem".
            *European Journal of Operations Research* 50, pp. 280-297. 1991.
        """
        self.generator = ecole.instance.CapacitatedFacilityLocationGenerator(
            n_customers, n_facilities, continuous_assignment, ratio,
            demand_interval, capacity_interval, fixed_cost_cste_interval,
            fixed_cost_scale_interval, rng.generator if rng is not None
            else rng
        )

    @property
    def n_customers(self) -> int:
        return self.generator.n_customers

    @property
    def n_facilities(self) -> int:
        return self.generator.n_facilities

    @property
    def continuous_assignment(self) -> bool:
        return self.generator.continuous_assignment

    @property
    def ratio(self) -> float:
        return self.generator.ratio

    @property
    def demand_interval(self) -> Tuple[int, int]:
        return self.generator.demand_interval

    @property
    def capacity_interval(self) -> Tuple[int, int]:
        return self.generator.capacity_interval

    @property
    def fixed_cost_cste_interval(self) -> Tuple[int, int]:
        return self.generator.fixed_cost_cste_interval

    @property
    def fixed_cost_scale_interval(self) -> Tuple[int, int]:
        return self.generator.fixed_cost_scale_interval
    
    @staticmethod
    def generate_instance(n_customers: int = 100, 
                          n_facilities: int = 100, 
                          continuous_assignment: bool = True, 
                          ratio: float = 5.0, 
                          demand_interval: Tuple[int, int] = (5, 36), 
                          capacity_interval: Tuple[int, int] = (10, 161), 
                          fixed_cost_cste_interval: Tuple[int, int] = (0, 91), 
                          fixed_cost_scale_interval: Tuple[int, int] = (100, 111), 
                          *, rng: RandomEngine) -> Model:
        """
        Generate a capacitated facility location MILP problem instance.

        The capacitated facility location assigns a number of customers to be served from a number of facilities.
        Not all facilities need to be opened.
        In fact, the problem is to minimized the sum of the fixed costs for each facilities and the sum of transportation
        costs for serving a given customer from a given facility.
        In a variant of the problem, the customers can be served from multiple facilities and the associated variables
        become [0,1] continuous.

        The sampling algorithm is described in [Cornuejols1991]_, but uniform sampling as been replaced by *integer*
        uniform sampling.

        Parameters
        ----------
        n_customers:
            The number of customers.
        n_facilities:
            The number of facilities.
        continuous_assignment:
            Whether variable for assigning a customer to a facility are binary or [0,1] continuous.
        ratio:
            After all sampling is performed, the capacities are scaled by `ratio * sum(demands) / sum(capacities)`.
        demand_interval:
            The customer demands are sampled independently as uniform integers in this interval [lower, upper[.
        capacity_interval:
            The facility capacities are sampled independently as uniform integers in this interval [lower, upper[.
        fixed_cost_cste_interval:
            The fixed costs are the sum of two terms.
            The first terms in the fixed costs for opening facilities are sampled independently as uniform integers
            in this interval [lower, upper[.
        fixed_cost_scale_interval:
            The fixed costs are the sum of two terms.
            The second terms in the fixed costs for opening facilities are sampled independently as uniform integers
            in this interval [lower, upper[ multiplied by the square root of their capacity prior to scaling.
            This second term reflects the economies of scale.
        rng:
            The random number generator used to peform all sampling.

        References
        ----------
        .. [Cornuejols1991]
            Cornuejols G, Sridharan R, Thizy J-M.
            "A Comparison of Heuristics and Relaxations for the Capacitated Plant Location Problem".
            *European Journal of Operations Research* 50, pp. 280-297. 1991.
        """
        return Model(
            ecole.instance
                 .CapacitatedFacilityLocationGenerator
                 .generate_instance(n_customers, n_facilities, 
                                    continuous_assignment, ratio,
                                    demand_interval, capacity_interval,
                                    fixed_cost_cste_interval,
                                    fixed_cost_scale_interval,
                                    rng.generator if rng is not None
                                    else rng)
        )
    
    def seed(self, seed: int) -> None:
        self.generator.seed(seed)

    def __iter__(self) -> "CapacitatedFacilityLocationGenerator":
        return self
    
    def __next__(self) -> Model:
        return Model(next(self.generator))
    
