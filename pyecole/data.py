import numbers
from .scip import Model
from .typing import DataFunction
import ecole.data
import pyecole
from typing import *

class NoneFunction(DataFunction):
    def __init__(self) -> None:
        self.func = ecole.data.NoneFunction()
    
    def before_reset(self, model: Model) -> None:
        """
        Do nothing.
        """
        self.func.before_reset(model.model)
    
    def extract(self, model: Model, done: bool) -> None:
        """
        Return None.
        """
        return self.func.extract(model.model, done)
    
class ConstantFunction(DataFunction):
    def __init__(self, arg0) -> None:
        self.func = ecole.data.ConstantFunction(arg0)

    def before_reset(self, model: Model) -> None:
        """
        Do nothing.
        """
        self.func.before_reset(model.model)
    
    def extract(self, model: Model, done: bool) -> object:
        """
        Return the constant.
        """
        return self.func.extract(model.model, done)

class VectorFunction(DataFunction):
    def __init__(self, *args) -> None:
        self.func = ecole.data.VectorFunction(*args)

    def before_reset(self, model: Model) -> None:
        """
        Call `before_reset()` on all data extraction functions.
        """
        self.func.before_reset(model.model)
    
    def extract(self, model: Model, done: bool) -> List[object]:
        """
        Return data from all functions as a tuple.
        """
        return self.func.extract(model.model, done)

class MapFunction(DataFunction):
    def __init__(self, **kwargs) -> None:
        self.func = ecole.data.MapFunction(**kwargs)

    def before_reset(self, model: Model) -> None:
        """
        Call `before_reset()` on all data extraction functions.
        """
        self.func.before_reset(model.model)
    
    def extract(self, model: Model, done: bool) -> Dict[str, object]:
        """
        Return data from all functions as a dict.
        """
        return self.func.extract(model.model, done)


def parse(something, default):
    """Recursively parse data function aggregates into their corresponding functions.

    For instance, vector of function are transformed into functions of vector, and similarily for
    maps, tuple, constants, etc.

    Parameters
    ----------
    something:
        Object to parse.
    default:
        Objet to return for when something is identified as asking for the environment specific default.

    Return
    ------
    data_func:
        A data extraction function to be used as an information, observation, or sometimes reward function.

    """
    if something == pyecole.Default:
        if default is None:
            raise ValueError("""Cannot parse "default" without a default value.""")
        return parse(default, None)
    elif something is None:
        return NoneFunction()
    elif isinstance(something, numbers.Number):
        return ConstantFunction(something)
    elif isinstance(something, (tuple, list)):
        return VectorFunction(*(parse(s, default) for s in something))
    elif isinstance(something, dict):
        return MapFunction(**{name: parse(s, default) for name, s in something.items()})
    else:
        return something
