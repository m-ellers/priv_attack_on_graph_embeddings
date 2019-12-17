from typing import List

import pandas as pd

import entities.graphs.graph as g


def compute_labels(node_list: List[int], graph: g.Graph, removed_node: int) -> pd.Series:
    """
    Computes labels for the classification task: labels are True for node in the node list if it is connected
     to :param removed_node in :param graph.
    :param node_list: list of nodes for which labels should be generated
    :param graph: the graph searched for links between nodes
    :param removed_node: node which is removed from the network in the attack and who's neighborhood structure should be
        recovered
    :return: labels
    """
    neighbors = graph.neighbours(removed_node)
    are_neighbors: List[bool] = list(map(lambda node: node in neighbors, node_list))
    return pd.Series(are_neighbors, index=node_list)
