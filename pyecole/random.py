import ecole
from typing import *

class RandomEngine:
    def __init__(self, value: Optional[int] = 5489) -> None:
        """
        Construct the pseudo-random number generator.
        """
        if value is not None:
            self.generator = ecole.RandomEngine(value)
    
    def seed(self, value: int = 5489) -> None:
        """
        Reinitialize the internal state of the random-number generator using 
        new seed value.
        """
        self.generator.seed(value)

    def discard(self, n: int) -> None:
        """
        Advance the internal state by `n` times. Equivalent to calling 
        `operator()` `n` times and discarding the result.
        """
        self.generator.discard(n)
    
    @property
    def max_seed(self) -> int:
        return self.generator.max_seed
    
    @property
    def min_seed(self) -> int:
        return self.generator.min_seed

def seed(val: int) -> None:
    """
    Seed the global source of randomness in Ecole.
    """
    ecole.seed(val)

def spawn_random_engine() -> RandomEngine:
    """
    Create new random generator deriving from global source of randomness.

    The global source of randomness is advance so two random engine created 
    successively have different states.
    """
    rng = RandomEngine(value=None)
    rng.generator = ecole.spawn_random_engine()
    return rng

