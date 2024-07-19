import ecole.typing
import math
from ..scip import Model
from ..typing import RewardFunction

class BaseRewardFunction(RewardFunction):
    def __init__(self, data: ecole.typing.RewardFunction) -> None:
        self.data = data

    def before_reset(self, model: Model) -> None:
        self.data.before_reset(model.model)

    def extract(self, model: Model, done: bool = False) -> float:
        return self.data.extract(model.model, done)
    
    def __add__(self, other) -> RewardFunction:
        return BaseRewardFunction(self.data.__add__(other))

    def __sub__(self, other) -> RewardFunction:
        return BaseRewardFunction(self.data.__sub__(other))

    def __mul__(self, other) -> RewardFunction:
        return BaseRewardFunction(self.data.__mul__(other))

    def __matmul__(self, other) -> RewardFunction:
        return BaseRewardFunction(self.data.__matmul__(other))

    def __truediv__(self, other) -> RewardFunction:
        return BaseRewardFunction(self.data.__truediv__(other))

    def __floordiv__(self, other) -> RewardFunction:
        return BaseRewardFunction(self.data.__floordiv__(other))

    def __mod__(self, other) -> RewardFunction:
        return BaseRewardFunction(self.data.__mod__(other))

    def __divmod__(self, other) -> RewardFunction:
        return BaseRewardFunction(self.data.__divmod__(other))

    def __pow__(self, other) -> RewardFunction:
        return BaseRewardFunction(self.data.__pow__(other))

    def __lshift__(self, other) -> RewardFunction:
        return BaseRewardFunction(self.data.__lshift__(other))

    def __rshift__(self, other) -> RewardFunction:
        return BaseRewardFunction(self.data.__rshift__(other))

    def __and__(self, other) -> RewardFunction:
        return BaseRewardFunction(self.data.__and__(other))

    def __xor__(self, other) -> RewardFunction:
        return BaseRewardFunction(self.data.__xor__(other))

    def __or__(self, other) -> RewardFunction:
        return BaseRewardFunction(self.data.__or__(other))

    def __radd__(self, other) -> RewardFunction:
        return BaseRewardFunction(self.data.__radd__(other))

    def __rsub__(self, other) -> RewardFunction:
        return BaseRewardFunction(self.data.__rsub__(other))

    def __rmul__(self, other) -> RewardFunction:
        return BaseRewardFunction(self.data.__rmul__(other))

    def __rmatmul__(self, other) -> RewardFunction:
        return BaseRewardFunction(self.data.__rmatmul__(other))

    def __rtruediv__(self, other) -> RewardFunction:
        return BaseRewardFunction(self.data.__rtruediv__(other))

    def __rfloordiv__(self, other) -> RewardFunction:
        return BaseRewardFunction(self.data.__rfloordiv__(other))

    def __rmod__(self, other) -> RewardFunction:
        return BaseRewardFunction(self.data.__rmod__(other))

    def __rdivmod__(self, other) -> RewardFunction:
        return BaseRewardFunction(self.data.__rdivmod__(other))

    def __rpow__(self, other) -> RewardFunction:
        return BaseRewardFunction(self.data.__rpow__(other))

    def __rlshift__(self, other) -> RewardFunction:
        return BaseRewardFunction(self.data.__rlshift__(other))

    def __rrshift__(self, other) -> RewardFunction:
        return BaseRewardFunction(self.data.__rrshift__(other))

    def __rand__(self, other) -> RewardFunction:
        return BaseRewardFunction(self.data.__rand__(other))

    def __rxor__(self, other) -> RewardFunction:
        return BaseRewardFunction(self.data.__rxor__(other))

    def __ror__(self, other) -> RewardFunction:
        return BaseRewardFunction(self.data.__ror__(other))

    def __neg__(self) -> RewardFunction:
        return BaseRewardFunction(self.data.__neg__())

    def __pos__(self) -> RewardFunction:
        return BaseRewardFunction(self.data.__pos__())

    def __abs__(self) -> RewardFunction:
        return BaseRewardFunction(self.data.__abs__())

    def __invert__(self) -> RewardFunction:
        return BaseRewardFunction(self.data.__invert__())

    def __int__(self) -> RewardFunction:
        return BaseRewardFunction(self.data.__int__())

    def __float__(self) -> RewardFunction:
        return BaseRewardFunction(self.data.__float__())

    def __complex__(self) -> RewardFunction:
        return BaseRewardFunction(self.data.__complex__())

    def __round__(self, ndigits=0) -> RewardFunction:
        return BaseRewardFunction(self.data.__round__(ndigits))

    def __trunc__(self) -> RewardFunction:
        return BaseRewardFunction(self.data.__trunc__())

    def __floor__(self) -> RewardFunction:
        return BaseRewardFunction(self.data.__floor__())

    def __ceil__(self) -> RewardFunction:
        return BaseRewardFunction(self.data.__ceil__())

    def exp(self) -> RewardFunction:
        return BaseRewardFunction(self.data.exp())

    def log(self, base=math.e) -> RewardFunction:
        return BaseRewardFunction(self.data.log(base))

    def log2(self) -> RewardFunction:
        return BaseRewardFunction(self.data.log2())

    def log10(self) -> RewardFunction:
        return BaseRewardFunction(self.data.log10())

    def sqrt(self) -> RewardFunction:
        return BaseRewardFunction(self.data.sqrt())

    def sin(self) -> RewardFunction:
        return BaseRewardFunction(self.data.sin())

    def cos(self) -> RewardFunction:
        return BaseRewardFunction(self.data.cos())

    def tan(self) -> RewardFunction:
        return BaseRewardFunction(self.data.tan())

    def asin(self) -> RewardFunction:
        return BaseRewardFunction(self.data.asin())

    def acos(self) -> RewardFunction:
        return BaseRewardFunction(self.data.acos())

    def atan(self) -> RewardFunction:
        return BaseRewardFunction(self.data.atan())

    def sinh(self) -> RewardFunction:
        return BaseRewardFunction(self.data.sinh())

    def cosh(self) -> RewardFunction:
        return BaseRewardFunction(self.data.cosh())

    def tanh(self) -> RewardFunction:
        return BaseRewardFunction(self.data.tanh())

    def asinh(self) -> RewardFunction:
        return BaseRewardFunction(self.data.asinh())

    def acosh(self) -> RewardFunction:
        return BaseRewardFunction(self.data.acosh())

    def atanh(self) -> RewardFunction:
        return BaseRewardFunction(self.data.atanh())

    def isfinite(self) -> RewardFunction:
        return BaseRewardFunction(self.data.isfinite())

    def isinf(self) -> RewardFunction:
        return BaseRewardFunction(self.data.isinf())

    def isnan(self) -> RewardFunction:
        return BaseRewardFunction(self.data.isnan())
    
    def apply(self, func) -> RewardFunction:
        return BaseRewardFunction(self.data.apply(func))
    
    def cumsum(self) -> RewardFunction:
        return BaseRewardFunction(self.data.cumsum())

