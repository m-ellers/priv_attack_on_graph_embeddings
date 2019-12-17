import functools
import multiprocessing
from typing import List, Dict

import scipy.spatial.distance
import pandas as pd
import numpy as np

import config
import entities.bins as b
import entities.graphs.graph as g
import entities.embeddings.embedding as e

import memory_access.memory_access as ma


def compute_cosine_distance_matrix(embedding: pd.DataFrame) -> pd.DataFrame:
    cos_matrix = scipy.spatial.distance.cdist(embedding.values, embedding.values, metric="cosine")
    return pd.DataFrame(cos_matrix, index=embedding.index, columns=embedding.index)


def compute_difference_matrix_by_difference(distance_matrix_one: pd.DataFrame,
                                            distance_matrix_two: pd.DataFrame) -> pd.DataFrame:
    assert (np.array_equal(distance_matrix_one.index.values, distance_matrix_two.index.values))
    assert (np.array_equal(distance_matrix_one.columns.values, distance_matrix_two.columns.values))
    assert (np.array_equal(distance_matrix_one.index.values, distance_matrix_one.columns.values))

    return distance_matrix_one - distance_matrix_two


def compute_bins(difference_matrix: pd.DataFrame, num_of_bins: int) -> b.Bins:
    assert (np.allclose(difference_matrix, difference_matrix.T))
    assert (len(difference_matrix.index) == len(difference_matrix.columns))

    diff_values = difference_matrix.values[np.triu_indices(len(difference_matrix), k=1)]

    _, bin_boundaries = pd.qcut(x=diff_values, q=num_of_bins, retbins=True)
    return b.Bins(bin_boundaries)


def compute_num_dist_values_in_bin_per_node(diff_matrix: pd.DataFrame, bins: b.Bins) -> pd.DataFrame:
    assert (np.allclose(diff_matrix, diff_matrix.T))  # is symmetric?
    # at least min boundary
    # noinspection PyTypeChecker
    assert (all(diff_matrix.values[np.triu_indices(len(diff_matrix), k=1)] >= bins.bin_boundaries[0]))
    # at most max boundary
    # noinspection PyTypeChecker
    assert (all(diff_matrix.values[np.triu_indices(len(diff_matrix), k=1)] <= bins.bin_boundaries[-1]))

    result_df = pd.DataFrame(index=diff_matrix.index, columns=list(range(bins.get_num_of_bins())), dtype=float)

    for row_index, row in diff_matrix.iterrows():
        result_df[0][row_index] = len(np.where((bins.bin_boundaries[0] <= row) & (row <= bins.bin_boundaries[1]))[0])
        for i in range(1, bins.get_num_of_bins()):
            result_df[i][row_index] = len(
                np.where((bins.bin_boundaries[i] < row) & (row <= bins.bin_boundaries[i + 1]))[0])

    return result_df


def compute_degree_column(node_list: List[int], graph: g.Graph):
    result_df = pd.DataFrame(0, index=node_list, columns=["deg"])
    for node in node_list:
        result_df["deg"][node] = graph.degree(node)
    return result_df


def compute_features_from_bin(diff_matrix: pd.DataFrame, bins: b.Bins, graph: g.Graph):
    features = compute_num_dist_values_in_bin_per_node(diff_matrix=diff_matrix, bins=bins)

    # normalise to percentages
    features /= len(diff_matrix)

    deg_column = compute_degree_column(node_list=diff_matrix.index, graph=graph)

    # normalise deg column (usually the smallest degree is 1, thus there is not much effect from moving the center)
    deg_column /= max(deg_column.values)

    # noinspection PyTypeChecker
    assert (all(features.index == deg_column.index))

    return features.join(deg_column)


def compute_features_from_distance_matrices(original_dm: pd.DataFrame, reduced_dm: pd.DataFrame,
                                            removed_nodes: List[int], num_of_bins: int,
                                            graph: g.Graph, embedding: e.Embedding,
                                            mem_acc: ma.MemoryAccess):
    if mem_acc.has_features(emb_func_name=str(embedding), graph_name=str(graph), removed_nodes=removed_nodes,
                            num_bins=num_of_bins):
        # features are already trained
        return

    original_dm = original_dm.drop(index=removed_nodes[-1], columns=removed_nodes[-1])

    # noinspection PyTypeChecker
    assert (all(original_dm.index == reduced_dm.index))
    # noinspection PyTypeChecker
    assert (all(original_dm.columns == reduced_dm.columns))

    diff_matrix = compute_difference_matrix_by_difference(distance_matrix_one=original_dm,
                                                          distance_matrix_two=reduced_dm)

    bins = compute_bins(difference_matrix=diff_matrix, num_of_bins=num_of_bins)
    features = compute_features_from_bin(diff_matrix=diff_matrix, bins=bins, graph=graph)

    mem_acc.save_features(features=features, emb_func_name=str(embedding), graph_name=str(graph),
                          removed_nodes=removed_nodes, num_bins=num_of_bins)
    return features


def compute_attack_and_training_features_for_one_attacked_node(attacked_node: int, trainig_nodes: List[int],
                                                               original_dm: pd.DataFrame,
                                                               graph: g.Graph, embedding: e.Embedding,
                                                               num_of_bins: int,
                                                               mem_acc: ma.MemoryAccess):
    dm_g = original_dm

    graph_p = graph.delete_node(attacked_node)
    del graph
    emb_g_p = mem_acc.load_embedding(emb_func_name=str(embedding), graph_name=str(graph_p),
                                     removed_nodes=[attacked_node])

    dm_g_p = compute_cosine_distance_matrix(emb_g_p)
    del emb_g_p
    compute_features_from_distance_matrices(original_dm=dm_g, reduced_dm=dm_g_p, removed_nodes=[attacked_node],
                                            num_of_bins=num_of_bins,
                                            graph=graph_p, embedding=embedding, mem_acc=mem_acc)
    del dm_g
    for second_rem_node in trainig_nodes:
        emb_g_pp = mem_acc.load_embedding(emb_func_name=str(embedding), graph_name=str(graph_p),
                                          removed_nodes=[attacked_node, second_rem_node])
        dm_g_pp = compute_cosine_distance_matrix(emb_g_pp)
        graph_pp = graph_p.delete_node(second_rem_node)

        compute_features_from_distance_matrices(original_dm=dm_g_p, reduced_dm=dm_g_pp,
                                                removed_nodes=[attacked_node, second_rem_node], num_of_bins=num_of_bins,
                                                graph=graph_pp, embedding=embedding, mem_acc=mem_acc)


def __pool_wrapper_compute_attack_and_training_features_for_one_attacked_node(
        dict_attacked_nodes_training_nodes: Dict[int, List[int]],
        original_dm: pd.DataFrame,
        graph: g.Graph, embedding: e.Embedding,
        num_of_bins: int,
        mem_acc: ma.MemoryAccess, attacked_node: int):
    compute_attack_and_training_features_for_one_attacked_node(
        attacked_node=attacked_node, trainig_nodes=dict_attacked_nodes_training_nodes[attacked_node],
        original_dm=original_dm, graph=graph, embedding=embedding,
        num_of_bins=num_of_bins, mem_acc=mem_acc)


def compute_attack_and_training_features_for_all_attacked_nodes(
        dict_attacked_nodes_training_nodes: Dict[int, List[int]],
        graph: g.Graph, embedding: e.Embedding,
        num_of_bins: int,
        mem_acc: ma.MemoryAccess) -> None:
    """
    Computes training features. It assumes that all embeddings are trained and can be loaded via mem_acc.
    :param dict_attacked_nodes_training_nodes: sampling of attacked and training nodes. Keys are the attacked nodes and
        the corresponding values the training nodes chosen for this attacked node
    :param graph: graph the features are crated from: used for naming and computation of node degrees
    :param embedding: type of the embeddings which should be processed: for data access of embeddings
    :param num_of_bins: number of bins used for the computation of training features
    :param mem_acc: obj for reading and writing embeddings and features
    """
    emb_g = mem_acc.load_embedding(emb_func_name=str(embedding), graph_name=str(graph),
                                   removed_nodes=[])
    original_dm = compute_cosine_distance_matrix(emb_g)

    func_p = functools.partial(__pool_wrapper_compute_attack_and_training_features_for_one_attacked_node,
                               dict_attacked_nodes_training_nodes, original_dm, graph, embedding, num_of_bins, mem_acc)

    with multiprocessing.Pool(min(config.NUM_CORES, len(dict_attacked_nodes_training_nodes))) as pool:
        for res in pool.imap(func_p, list(dict_attacked_nodes_training_nodes.keys())):
            pass
