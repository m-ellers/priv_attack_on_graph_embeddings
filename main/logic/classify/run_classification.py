import functools
import multiprocessing
from typing import List, Dict
import sklearn.base as sk_base
import pandas as pd

import config
import entities.graphs.graph as g
import entities.embeddings.embedding as e

import logic.classify.train_classifier as tc
import logic.classify.evaluate_classifier as ec

import memory_access.memory_access as ma


def run_classification_for_one_node(attacked_node: int, training_nodes: List[int],
                                    classifier: sk_base.ClassifierMixin,
                                    graph: g.Graph, embedding: e.Embedding, num_bins: int,
                                    mem_acc: ma.MemoryAccess) -> pd.DataFrame:
    graph_p = graph.delete_node(attacked_node)
    model = tc.train_classifier(attacked_node=attacked_node,
                                training_nodes=training_nodes,
                                graph_p=graph_p,
                                embedding=embedding,
                                num_bins=num_bins,
                                classifier=classifier,
                                mem_acc=mem_acc)

    evaluation = ec.evaluate_classifier(trained_classifier=model, attacked_node=attacked_node,
                                        training_nodes=training_nodes,
                                        graph=graph,
                                        embedding=embedding, num_bins=num_bins,
                                        mem_acc=mem_acc)

    return evaluation


def __pool_wrapper_run_classification_for_one_node(dict_attacked_nodes_training_nodes: Dict[int, List[int]],
                                                   classifier: sk_base.ClassifierMixin,
                                                   graph: g.Graph, embedding: e.Embedding, num_bins: int,
                                                   mem_acc: ma.MemoryAccess, attacked_node: int) -> pd.DataFrame:
    return run_classification_for_one_node(attacked_node=attacked_node,
                                           training_nodes=dict_attacked_nodes_training_nodes[attacked_node],
                                           classifier=classifier, graph=graph, embedding=embedding, num_bins=num_bins,
                                           mem_acc=mem_acc)


def run_classification(dict_attacked_nodes_training_nodes: Dict[int, List[int]],
                       classifier: sk_base.ClassifierMixin, graph: g.Graph, embedding: e.Embedding,
                       num_bins: int, mem_acc: ma.MemoryAccess) -> (pd.DataFrame, pd.DataFrame):
    """
    Trains a classifier and evaluates it
    :param dict_attacked_nodes_training_nodes: keys: attacked nodes, values: list of training_nodes
    :param classifier: classifier which is used for training the model
    :param graph: graph the features are created from
    :param embedding: embedding used for feature createion
    :param num_bins: number of bins used for feature creation
    :param mem_acc: obj for memory access
    :return: evaluation results per attacked node and aggregated for all attacked nodes
    """
    evaluations_df: pd.DataFrame
    # load results if available
    if mem_acc.has_test_results_per_node(dict_attack_train_nodes=dict_attacked_nodes_training_nodes,
                                         classifier_name=str(classifier), emb_func_name=str(embedding),
                                         graph_name=str(graph), num_bins=num_bins):
        evaluations_df = mem_acc.load_test_results_per_node(
            dict_attack_train_nodes=dict_attacked_nodes_training_nodes,
            classifier_name=str(classifier),
            emb_func_name=str(embedding),
            graph_name=str(graph), num_bins=num_bins)
    else:
        evaluations = []
        func_p = functools.partial(__pool_wrapper_run_classification_for_one_node, dict_attacked_nodes_training_nodes,
                                   classifier, graph, embedding, num_bins, mem_acc)

        with multiprocessing.Pool(min(config.NUM_CORES, len(dict_attacked_nodes_training_nodes))) as pool:
            for evaluation in pool.imap(func_p, list(dict_attacked_nodes_training_nodes.keys())):
                evaluations.append(evaluation)

        evaluations_df = pd.concat(evaluations, axis=1)

        mem_acc.save_test_results_per_node(results_per_node=evaluations_df,
                                           dict_attack_train_nodes=dict_attacked_nodes_training_nodes,
                                           classifier_name=str(classifier), emb_func_name=str(embedding),
                                           graph_name=str(graph), num_bins=num_bins)

    if mem_acc.has_aggregated_test_results(dict_attack_train_nodes=dict_attacked_nodes_training_nodes,
                                           classifier_name=str(classifier), emb_func_name=str(embedding),
                                           graph_name=str(graph), num_bins=num_bins):
        aggregated_evaluations = mem_acc.load_aggregated_test_results(
            dict_attack_train_nodes=dict_attacked_nodes_training_nodes,
            classifier_name=str(classifier), emb_func_name=str(embedding),
            graph_name=str(graph), num_bins=num_bins)
    else:
        aggregated_evaluations = ec.aggregate_evaluations(evaluations=evaluations_df)
        mem_acc.save_aggregated_test_results(results_per_node=aggregated_evaluations,
                                             dict_attack_train_nodes=dict_attacked_nodes_training_nodes,
                                             classifier_name=str(classifier), emb_func_name=str(embedding),
                                             graph_name=str(graph), num_bins=num_bins)

    return evaluations_df, aggregated_evaluations
