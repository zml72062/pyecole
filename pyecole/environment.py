"""Ecole collection of environments."""

import ecole
import os
import pyecole
import pyecole.observation
import pyecole.reward
import pyecole.dynamics
from .scip import Model
from .data import parse
from .random import RandomEngine
from typing import *
from .typing import DataFunction, Dynamics


class Environment:
    """Ecole Partially Observable Markov Decision Process (POMDP).

    Similar to OpenAI Gym, environments represent the task that an agent is supposed to solve.
    For maximum customizability, different components are composed/orchestrated in this class.
    """

    __Dynamics__ = None
    __DefaultObservationFunction__ = pyecole.observation.Nothing
    __DefaultRewardFunction__ = pyecole.reward.IsDone
    __DefaultInformationFunction__ = pyecole.observation.Nothing
    reward_function: DataFunction
    observation_function: DataFunction
    information_function: DataFunction
    scip_params: Dict[str, Union[bool, int, float, str]]
    dynamics: Dynamics
    can_transition: bool
    rng: RandomEngine

    def __init__(
        self,
        observation_function=pyecole.Default,
        reward_function=pyecole.Default,
        information_function=pyecole.Default,
        scip_params: Optional[Dict[str, Union[bool, int, float, str]]]=None,
        **dynamics_kwargs
    ) -> None:
        """Create a new environment object.

        Parameters
        ----------
        observation_function:
            An object of type `pyecole.observation.ObservationFunction` used to customize the
            observation returned by `reset` and `step`.
        reward_function:
            An object of type `pyecole.reward.RewardFunction` used to customize the reward
            returned by `reset` and `step`.
        information_function:
            An object of type `pyecole.typing.InformationFunction` used to customize the
            additional information returned by `reset` and `step`.
        scip_params:
            Parameters set on the underlying `pyecole.scip.Model` at the start of every episode.
        **dynamics_kwargs:
            Other arguments are passed to the constructor of the `pyecole.typing.Dynamics`.

        """

        self.reward_function = parse(reward_function, self.__DefaultRewardFunction__())
        self.observation_function = parse(
            observation_function, self.__DefaultObservationFunction__()
        )
        self.information_function = parse(
            information_function, self.__DefaultInformationFunction__()
        )
        self.scip_params = scip_params if scip_params is not None else {}
        self.model = None
        self.dynamics = self.__Dynamics__(**dynamics_kwargs)
        self.can_transition = False
        self.rng = pyecole.spawn_random_engine()

    def reset(self, instance: Union[Model, os.PathLike]):
        """Start a new episode.

        This method brings the environment to a new initial state, *i.e.* starts a new
        episode.
        The method can be called at any point in time.

        Parameters
        ----------
        instance:
            The combinatorial optimization problem to tackle during the newly started
            episode.
            Either a file path to an instance that can be read by SCIP, or a `Model` whose problem
            definition data will be copied.

        Returns
        -------
        observation:
            The observation extracted from the initial state.
            Typically used to take the next action.
        action_set:
            An optional subset that defines which actions are accepted in the next transition.
            For some environment, the action set may change at every transition.
        reward_offset:
            An offset on the total cumulated reward, a.k.a. the initial reward.
            This reward does not impact learning (as no action has yet been taken) but can nonetheless
            be used for evaluation purposes. For example, in the total cumulated reward of an episode
            one may want to account for computations that happened during :py`reset`
            (*e.g.* computation time, number of LP iteration in presolving...).
        done:
            A boolean flag indicating whether the current state is terminal.
            If this flag is true, then the current episode is finished, and `step`
            cannot be called any more.
        info:
            A collection of environment specific information about the transition.
            This is not necessary for the control problem, but is useful to gain
            insights about the environment.

        """
        self.can_transition = True
        try:
            if isinstance(instance, Model):
                self.model = instance.copy_orig()
            else:
                self.model = Model.from_file(instance)
            self.model.set_params(self.scip_params)

            self.dynamics.set_dynamics_random_state(self.model, self.rng)

            # Reset data extraction functions
            self.reward_function.before_reset(self.model)
            self.observation_function.before_reset(self.model)
            self.information_function.before_reset(self.model)

            # Place the environment in its initial state
            done, action_set = self.dynamics.reset_dynamics(self.model)
            self.can_transition = not done

            # Extract additional information to be returned by reset
            reward_offset = self.reward_function.extract(self.model, done)
            if not done:
                observation = self.observation_function.extract(self.model, done)
            else:
                observation = None
            information = self.information_function.extract(self.model, done)

            return observation, action_set, reward_offset, done, information
        except Exception as e:
            self.can_transition = False
            raise e

    def step(self, action):
        """Transition from one state to another.

        This method takes a user action to transition from the current state to the
        next.
        The method **cannot** be called if the environment has not been reset since its
        instantiation or since a terminal state has been reached.

        Parameters
        ----------
        action:
            The action to take in as part of the Markov Decision Process.
            If an action set has been given in the latest call (inluding calls to
            :meth:`reset`), then the action **must** comply with the action set.

        Returns
        -------
        observation:
            The observation extracted from the initial state.
            Typically used to take the next action.
        action_set:
            An optional subset that defines which actions are accepted in the next transition.
            For some environment, the action set may change at every transition.
        reward:
            A real number to use for reinforcement learning.
        done:
            A boolean flag indicating whether the current state is terminal.
            If this flag is true, then the current episode is finished, and :meth:`step`
            cannot be called any more.
        info:
            A collection of environment specific information about the transition.
            This is not necessary for the control problem, but is useful to gain
            insights about the environment.

        """
        if not self.can_transition:
            raise ecole.MarkovError("Environment need to be reset.")

        try:
            # Transition the environment to the next state
            done, action_set = self.dynamics.step_dynamics(self.model, action)
            self.can_transition = not done

            # Extract additional information to be returned by step
            reward = self.reward_function.extract(self.model, done)
            if not done:
                observation = self.observation_function.extract(self.model, done)
            else:
                observation = None
            information = self.information_function.extract(self.model, done)

            return observation, action_set, reward, done, information
        except Exception as e:
            self.can_transition = False
            raise e

    def seed(self, value: int) -> None:
        """Set the random seed of the environment.

        The random seed is used to seed the environment :py:class:`~ecole.RandomGenerator`.
        At every call to :py:meth:`reset`, the random generator is used to create new seeds
        for the solver.
        Setting the seed once will ensure determinism for the next trajectories.
        By default, the random generator is initialized by the
        `random <https://docs.python.org/library/random.html>`_ module.
        """
        self.rng.seed(value)


class Branching(Environment):
    __Dynamics__ = pyecole.dynamics.BranchingDynamics
    __DefaultObservationFunction__ = pyecole.observation.NodeBipartite


class Configuring(Environment):
    __Dynamics__ = pyecole.dynamics.ConfiguringDynamics


class PrimalSearch(Environment):
    __Dynamics__ = pyecole.dynamics.PrimalSearchDynamics
    __DefaultObservationFunction__ = pyecole.observation.NodeBipartite

