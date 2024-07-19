import ecole.observation
from ..scip import Model
from ..typing import ObservationFunction

class Nothing(ObservationFunction):
    """
    Always return None.
    """
    def __init__(self):
        self.func = ecole.observation.Nothing()

    def before_reset(self, model: Model) -> None:
        self.func.before_reset(model.model)

    def extract(self, model: Model, done: bool):
        return self.func.extract(model.model, done)

