import ecole.instance
from ..typing import InstanceGenerator
from ..random import RandomEngine
from ..scip.model import Model
from enum import Enum

class IndependentSetGenerator(InstanceGenerator):
    class GraphType(Enum):
        barabasi_albert = "barabasi_albert"
        erdos_renyi = "erdos_renyi"

    def __init__(self, n_nodes: int = 500, 
                 graph_type: GraphType = GraphType.barabasi_albert,
                 edge_probability: float = 0.25, 
                 affinity: int = 4, 
                 rng: RandomEngine = None) -> None:
        """
        Generate an independent set MILP problem instance.

        Given an undireted graph, the problem is to find a maximum subset of nodes such that no pair of nodes are
        connected. There are one variable per node in the underlying graph. Instead of adding one constraint per edge, a
        greedy algorithm is run to replace these inequalities when clique is found. The maximization problem is
        unwheighted, that is all objective coefficients are equal to one.

        The problem are generated using the procedure from [Bergman2016]_, and the graphs are sampled following
        [Erdos1959]_ and [Barabasi1999]_.

        Parameters
        ----------
        n_nodes:
            The number of nodes in the graph, and therefore of variable.
        graph_type:
            The method used in which to generate graphs.
            One of ``"barabasi_albert"`` or ``"erdos_renyi"``.
        edge_probability:
            The probability of generating each edge.
            This parameter must be in the range [0, 1].
            This parameter will only be used if ``graph_type == "erdos_renyi"``.
        affinity:
            The number of nodes each new node will be attached to, in the sampling scheme.
            This parameter must be an integer >= 1.
            This parameter will only be used if ``graph_type == "barabasi_albert"``.
        rng:
            The random number generator used to peform all sampling.

        References
        ----------
            .. [Bergman2016]
                David Bergman, Andre A. Cire, Willem-Jan Van Hoeve, and John Hooker.
                "Decision diagrams for optimization", Section 4.6.4.
                *Springer International Publishing*, 2016.
            .. [Erdos1959]
                Paul Erdos and Alfréd Renyi.
                "On Random Graph"
                *Publicationes Mathematicae*, pp. 290-297, 1959.
            .. [Barabasi1999]
                Albert-László Barabási and Réka Albert.
                "Emergence of scaling in random networks"
                *Science* vol. 286, num. 5439, pp. 509-512, 1999.
        """
        self.generator = ecole.instance.IndependentSetGenerator(
            n_nodes, ecole.instance.IndependentSetGenerator.GraphType(graph_type.value),
            edge_probability, affinity, 
            rng.generator if rng is not None else rng
        )

    @property
    def n_nodes(self) -> int:
        return self.generator.n_nodes

    @property
    def graph_type(self) -> GraphType:
        return self.GraphType(self.generator.graph_type.name)

    @property
    def edge_probability(self) -> float:
        return self.generator.edge_probability

    @property
    def affinity(self) -> int:
        return self.generator.affinity

    @staticmethod
    def generate_instance(n_nodes: int = 500, 
                          graph_type: GraphType = GraphType.barabasi_albert,
                          edge_probability: float = 0.25, 
                          affinity: int = 4, 
                          *, rng: RandomEngine) -> Model:
        """
        Generate an independent set MILP problem instance.

        Given an undireted graph, the problem is to find a maximum subset of nodes such that no pair of nodes are
        connected. There are one variable per node in the underlying graph. Instead of adding one constraint per edge, a
        greedy algorithm is run to replace these inequalities when clique is found. The maximization problem is
        unwheighted, that is all objective coefficients are equal to one.

        The problem are generated using the procedure from [Bergman2016]_, and the graphs are sampled following
        [Erdos1959]_ and [Barabasi1999]_.

        Parameters
        ----------
        n_nodes:
            The number of nodes in the graph, and therefore of variable.
        graph_type:
            The method used in which to generate graphs.
            One of ``"barabasi_albert"`` or ``"erdos_renyi"``.
        edge_probability:
            The probability of generating each edge.
            This parameter must be in the range [0, 1].
            This parameter will only be used if ``graph_type == "erdos_renyi"``.
        affinity:
            The number of nodes each new node will be attached to, in the sampling scheme.
            This parameter must be an integer >= 1.
            This parameter will only be used if ``graph_type == "barabasi_albert"``.
        rng:
            The random number generator used to peform all sampling.

        References
        ----------
            .. [Bergman2016]
                David Bergman, Andre A. Cire, Willem-Jan Van Hoeve, and John Hooker.
                "Decision diagrams for optimization", Section 4.6.4.
                *Springer International Publishing*, 2016.
            .. [Erdos1959]
                Paul Erdos and Alfréd Renyi.
                "On Random Graph"
                *Publicationes Mathematicae*, pp. 290-297, 1959.
            .. [Barabasi1999]
                Albert-László Barabási and Réka Albert.
                "Emergence of scaling in random networks"
                *Science* vol. 286, num. 5439, pp. 509-512, 1999.
        """
        return Model(
            ecole.instance
                 .IndependentSetGenerator
                 .generate_instance(n_nodes,
                                    ecole.instance
                                         .IndependentSetGenerator
                                         .GraphType(graph_type.value),
                                    edge_probability, affinity,
                                    rng.generator if rng is not None
                                    else rng)
        )
    
    def seed(self, seed: int) -> None:
        self.generator.seed(seed)

    def __iter__(self) -> "IndependentSetGenerator":
        return self
    
    def __next__(self) -> Model:
        return Model(next(self.generator))
    
