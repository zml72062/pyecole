import os
import ecole.scip
from typing import *

class Model:
    def __init__(self, model: ecole.scip.Model) -> None:
        self.model = model

    def as_pyscipopt(self) -> object:
        return self.model.as_pyscipopt()
    
    def copy_orig(self) -> "Model":
        return Model(self.model.copy_orig())
    
    def disable_cuts(self) -> None:
        self.model.disable_cuts()

    def disable_presolve(self) -> None:
        self.model.disable_presolve()

    @property
    def primal_bound(self) -> float:
        return self.model.primal_bound

    @property
    def dual_bound(self) -> float:
        return self.model.dual_bound

    @property
    def name(self) -> str:
        return self.model.name
    
    @property
    def is_solved(self) -> bool:
        return self.model.is_solved
    
    @property
    def stage(self) -> ecole.scip.Stage:
        return self.model.stage
    
    @staticmethod
    def from_file(filepath: os.PathLike) -> "Model":
        return Model(ecole.scip.Model.from_file(filepath))
    
    @staticmethod
    def from_pyscipopt(model: object) -> "Model":
        return Model(ecole.scip.Model.from_pyscipopt(model))
    
    @staticmethod
    def prob_basic(name: str = "Model") -> "Model":
        return Model(ecole.scip.Model.prob_basic(name))
    
    def get_param(self, name: str) -> Union[bool, int, float, str]:
        return self.model.get_param(name)
    
    def get_params(self) -> Dict[str, Union[bool, int, float, str]]:
        return self.model.get_params()

    def set_param(self, name: str, value: Union[bool, int, float, str]) -> None:
        self.model.set_param(name, value)

    def set_params(self, 
                   name_values: Dict[str, Union[bool, int, float, str]]) -> None:
        self.model.set_params(name_values)
    
    def presolve(self) -> None:
        self.model.presolve()

    def solve(self) -> None:
        self.model.solve()

    def set_messagehdlr_quiet(self, quiet: bool) -> None:
        self.model.set_messagehdlr_quiet(quiet)

    def transform_prob(self) -> None:
        self.model.transform_prob()

    def write_problem(self, filepath: os.PathLike) -> None:
        self.model.write_problem(filepath)
    
