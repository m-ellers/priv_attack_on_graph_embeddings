from typing import List, Dict

import entities.graphs.graph as g
import entities.embeddings.embedding as emb

import memory_access.memory_access as ma


def train_embeddings_for_each_attacked_node(graph: g.Graph, embedding: emb.Embedding,
                                            dict_attacked_nodes_training_nodes: Dict[int, List[int]],
                                            mem_acc: ma.MemoryAccess) -> None:
    """
    Trains all embeddings for experiments.
    :param graph: original graph with all nodes
    :param embedding: embedding which should be used
    :param dict_attacked_nodes_training_nodes: sampling of attacked and trainig nodes
    :param mem_acc: obj for saving and loading embeddings
    """
    for attacked_node in dict_attacked_nodes_training_nodes:
        train_embeddings(graph=graph, embedding=embedding, attacked_node=attacked_node,
                         training_second_removed_nodes=dict_attacked_nodes_training_nodes[attacked_node],
                         mem_acc=mem_acc)


def train_embeddings(graph: g.Graph, embedding: emb.Embedding,
                     attacked_node: int, training_second_removed_nodes: List[int],
                     mem_acc: ma.MemoryAccess):
    embedding.train_embedding(graph=graph, memory_access=mem_acc, removed_nodes=[])

    graph_p = graph.delete_node(attacked_node)
    embedding.train_embedding(graph=graph_p, memory_access=mem_acc, removed_nodes=[attacked_node])

    for second_rem_node in training_second_removed_nodes:
        graph_pp = graph_p.delete_node(second_rem_node)
        embedding.train_embedding(graph=graph_pp, memory_access=mem_acc, removed_nodes=[attacked_node, second_rem_node])
