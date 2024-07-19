import ecole.instance
from ..typing import InstanceGenerator
from ..random import RandomEngine
from ..scip.model import Model

class CombinatorialAuctionGenerator(InstanceGenerator):
    def __init__(self, n_items: int = 100, 
                 n_bids: int = 500, 
                 min_value: int = 1, 
                 max_value: int = 100, 
                 value_deviation: float = 0.5, 
                 add_item_prob: float = 0.65, 
                 max_n_sub_bids: int = 5, 
                 additivity: float = 0.2, 
                 budget_factor: float = 1.5, 
                 resale_factor: float = 0.5, 
                 integers: bool = False, 
                 warnings: bool = False, 
                 rng: RandomEngine = None) -> None:
        """
        Generate a combinatorial auction MILP problem instance.

        This method generates an instance of a combinatorial auction problem based on the
        specified parameters and returns it as an ecole model.

        Algorithm described in [LeytonBrown2000]_.

        Parameters
        ----------
        n_items:
            The number of items.
        n_bids:
            The number of bids.
        min_value:
            The minimum resale value for an item.
        max_value:
            The maximum resale value for an item.
        value_deviation:
            The deviation allowed for each bidder's private value of an item, relative from max_value.
        add_item_prob:
            The probability of adding a new item to an existing bundle.
            This parameters must be in the range [0,1].
        max_n_sub_bids:
            The maximum number of substitutable bids per bidder (+1 gives the maximum number of bids per bidder).
        additivity:
            Additivity parameter for bundle prices. Note that additivity < 0 gives sub-additive bids, while
            additivity > 0 gives super-additive bids.
        budget_factor:
            The budget factor for each bidder, relative to their initial bid's price.
        resale_factor:
            The resale factor for each bidder, relative to their initial bid's resale value.
        integers:
            Determines if the bid prices should be integral.
        warnings:
            Determines if warnings should be printed when invalid bundles are skipped in instance generation.
        rng:
            The random number generator used to peform all sampling.

        References
        ----------
        .. [LeytonBrown2000]
            Kevin Leyton-Brown, Mark Pearson, and Yoav Shoham.
            "Towards a universal test suite for combinatorial auction algorithms".
            *Proceedings of ACM Conference on Electronic Commerce* (EC01) pp. 66-76.
            Section 4.3., the 'arbitrary' scheme. 2000.
        """
        self.generator = ecole.instance.CombinatorialAuctionGenerator(
            n_items, n_bids, min_value, max_value, value_deviation,
            add_item_prob, max_n_sub_bids, additivity, budget_factor,
            resale_factor, integers, warnings, 
            rng.generator if rng is not None else rng
        )

    @property
    def n_items(self) -> int:
        return self.generator.n_items
        
    @property
    def n_bids(self) -> int:
        return self.generator.n_bids
        
    @property
    def min_value(self) -> int:
        return self.generator.min_value
        
    @property
    def max_value(self) -> int:
        return self.generator.max_value
        
    @property
    def value_deviation(self) -> float:
        return self.generator.value_deviation
        
    @property
    def add_item_prob(self) -> float:
        return self.generator.add_item_prob
        
    @property
    def max_n_sub_bids(self) -> int:
        return self.generator.max_n_sub_bids
        
    @property
    def additivity(self) -> float:
        return self.generator.additivity
        
    @property
    def budget_factor(self) -> float:
        return self.generator.budget_factor
        
    @property
    def resale_factor(self) -> float:
        return self.generator.resale_factor
        
    @property
    def integers(self) -> bool:
        return self.generator.integers
        
    @property
    def warnings(self) -> bool:
        return self.generator.warnings

    @staticmethod
    def generate_instance(n_items: int = 100, 
                          n_bids: int = 500, 
                          min_value: int = 1, 
                          max_value: int = 100, 
                          value_deviation: float = 0.5, 
                          add_item_prob: float = 0.65, 
                          max_n_sub_bids: int = 5, 
                          additivity: float = 0.2, 
                          budget_factor: float = 1.5, 
                          resale_factor: float = 0.5, 
                          integers: bool = False, 
                          warnings: bool = False, 
                          *, rng: RandomEngine = None) -> Model:
        """
        Generate a combinatorial auction MILP problem instance.

        This method generates an instance of a combinatorial auction problem based on the
        specified parameters and returns it as an ecole model.

        Algorithm described in [LeytonBrown2000]_.

        Parameters
        ----------
        n_items:
            The number of items.
        n_bids:
            The number of bids.
        min_value:
            The minimum resale value for an item.
        max_value:
            The maximum resale value for an item.
        value_deviation:
            The deviation allowed for each bidder's private value of an item, relative from max_value.
        add_item_prob:
            The probability of adding a new item to an existing bundle.
            This parameters must be in the range [0,1].
        max_n_sub_bids:
            The maximum number of substitutable bids per bidder (+1 gives the maximum number of bids per bidder).
        additivity:
            Additivity parameter for bundle prices. Note that additivity < 0 gives sub-additive bids, while
            additivity > 0 gives super-additive bids.
        budget_factor:
            The budget factor for each bidder, relative to their initial bid's price.
        resale_factor:
            The resale factor for each bidder, relative to their initial bid's resale value.
        integers:
            Determines if the bid prices should be integral.
        warnings:
            Determines if warnings should be printed when invalid bundles are skipped in instance generation.
        rng:
            The random number generator used to peform all sampling.

        References
        ----------
        .. [LeytonBrown2000]
            Kevin Leyton-Brown, Mark Pearson, and Yoav Shoham.
            "Towards a universal test suite for combinatorial auction algorithms".
            *Proceedings of ACM Conference on Electronic Commerce* (EC01) pp. 66-76.
            Section 4.3., the 'arbitrary' scheme. 2000.
        """
        return Model(
            ecole.instance
                 .CombinatorialAuctionGenerator
                 .generate_instance(n_items, n_bids, min_value, max_value,
                                    value_deviation, add_item_prob,
                                    max_n_sub_bids, additivity, budget_factor,
                                    resale_factor, integers, warnings, 
                                    rng.generator if rng is not None else rng)
        )
    
    def seed(self, seed: int) -> None:
        self.generator.seed(seed)

    def __iter__(self) -> "CombinatorialAuctionGenerator":
        return self
    
    def __next__(self) -> Model:
        return Model(next(self.generator))
    
