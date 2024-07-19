import numpy as np
from typing import *

class coo_matrix:
    @property
    def indices(self) -> np.ndarray:
        raise NotImplementedError()
    
    @property
    def values(self) -> np.ndarray:
        raise NotImplementedError()
    
    @property
    def shape(self) -> List[int]:
        raise NotImplementedError()
    
    @property
    def nnz(self) -> int:
        raise NotImplementedError()
    
