from typing import Dict, List

import entities.parameters as p
import entities.graphs.graph as g

import logic.node_sampling.sampler as sampler
import logic.train_embeddings.train_embedding as te
import logic.compute_features.compute_features as cf
import logic.classify.run_classification as rc
import memory_access.memory_access as ma


def sample_nodes(graph: g.Graph,
                 num_attacked_nodes_pre_deg_level: int,
                 num_training_nodes: int) -> Dict[int, List[int]]:
    """
    Samples attacked nodes from the given graph with low, high and medium degree.
    And for each computes training nodes which are to be removed from the graph.
    :param graph: original graph from which the attacked node is removed
    :param num_attacked_nodes_pre_deg_level: number of different attacked nodes:
    number_of_attacked_nodes_pre_deg_level * 3
    (experiments for attacked low degree nodes, middle degree nodes and high degree nodes)
    :param num_training_nodes: number of second nodes which should be removed from the network to train the
    classifier
    :return: dictionary keys: attacked nodes, values: training second removed nodes
    """

    return sampler.sample_attacked_and_training_nodes(
        graph=graph,
        number_of_attacked_nodes=num_attacked_nodes_pre_deg_level,
        number_of_training_nodes=num_training_nodes)


def run(parameters: p.Parameters, memory_access: ma.MemoryAccess):
    """
    Pipeline to perform the whole experiment. Results of each step are saved in memory via the memory_access class.
    Thus, steps can be performed separately.
    :param parameters: contains all parameters of the experiments
    :param memory_access: interface/class for memory access
    """

    print("Train embeddings")
    te.train_embeddings_for_each_attacked_node(
        graph=parameters.graph, embedding=parameters.embedding,
        dict_attacked_nodes_training_nodes=parameters.dict_attacked_nodes_training_nodes,
        mem_acc=memory_access)

    print("Compute features")
    cf.compute_attack_and_training_features_for_all_attacked_nodes(
        dict_attacked_nodes_training_nodes=parameters.dict_attacked_nodes_training_nodes,
        graph=parameters.graph, embedding=parameters.embedding, num_of_bins=parameters.num_of_bins,
        mem_acc=memory_access)

    print("Train and Evaluate Classifier")
    evaluations_df, aggregated_evaluations = rc.run_classification(
        dict_attacked_nodes_training_nodes=parameters.dict_attacked_nodes_training_nodes,
        classifier=parameters.classifier, graph=parameters.graph,
        embedding=parameters.embedding,
        num_bins=parameters.num_of_bins, mem_acc=memory_access)

    print("Performance per attacked node:")
    print(evaluations_df)
    print("Aggregated performance:")
    print(aggregated_evaluations)

    return evaluations_df, aggregated_evaluations
