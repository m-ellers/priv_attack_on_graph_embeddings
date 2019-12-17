import abc
from typing import List

import entities.graphs.graph as g
import memory_access.memory_access as ma


class Embedding(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def train_embedding(self, graph: g.Graph, memory_access: ma.MemoryAccess,
                        removed_nodes: List[int]):
        assert (all(node not in graph.nodes() for node in removed_nodes))

    @abc.abstractmethod
    def __str__(self) -> str:
        pass

    def short_name(self) -> str:
        pass
