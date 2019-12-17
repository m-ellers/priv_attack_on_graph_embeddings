import numpy as np
from typing import List, Dict

import entities.graphs.graph as gc


def sample_attacked_and_training_nodes(graph: gc.Graph, number_of_attacked_nodes,
                                       number_of_training_nodes) -> Dict[int, List[int]]:
    """
    Samples attacked nodes from the given graph with low, high and medium degree.
    And for each computes training nodes which are to be removed from the graph.
    :param graph: original graph from which the attacked node is removed
    :param number_of_attacked_nodes: number of different experiments that is performed: number_of_attacked_nodes*3
    ( experiments for attacked low degree nodes, middle degree nodes and high degree nodes)
    :param number_of_training_nodes: number of second nodes which should be removed from the network to train the
    classifier
    :return: dictionary keys: attacked nodes, values: training second removed nodes
    """
    list_nodes_to_test = sample_low_avg_high_degree_nodes(graph=graph, quantity=number_of_attacked_nodes)

    dict_attacked_training_nodes = {}
    for node in list_nodes_to_test:
        reduced_graph = graph.delete_node(node)  # so the attacked node is not sampled again
        training_nodes = sample_without_graph_splitting_nodes(graph=reduced_graph, quantity=number_of_training_nodes)
        dict_attacked_training_nodes[node] = training_nodes

    return dict_attacked_training_nodes


def sample_low_avg_high_degree_nodes(graph: gc.Graph, quantity: int, init_range: int = 2) -> List[int]:
    """
    Smples nodes from the input graph with high, middle and low degree
    :param graph: input graph the nodes should be sampled from
    :param quantity: number of nodes that should be sampled per degree
    :param init_range: defines initial range of degrees around the calculated low/medium/high degree from which nodes
    should be sampled. If there are not enough nodes in this range the available nodes are sampled
    and the range is successively increased to include more node degrees
    :return: node names that should be sampled
    """

    dict_deg_to_node_list = {}
    for node in graph.nodes():
        if not graph.splits_graph(node):
            deg = graph.degree(node)
            dict_deg_to_node_list[deg] = dict_deg_to_node_list.get(deg, []) + [node]

    min_deg = min(dict_deg_to_node_list.keys())
    max_deg = max(dict_deg_to_node_list.keys())
    middle_deg: int = int(round((max_deg + min_deg) / 2))

    max_deg_node_sample = get_sample(dict_deg_to_node_list=dict_deg_to_node_list,
                                     max_deg=max_deg, min_deg=min_deg,
                                     sampling_node_deg_center=max_deg, sampling_node_deg_radius=init_range,
                                     quantity=quantity, neg_list=[])
    min_deg_node_sample = get_sample(dict_deg_to_node_list=dict_deg_to_node_list,
                                     max_deg=max_deg, min_deg=min_deg,
                                     sampling_node_deg_center=min_deg, sampling_node_deg_radius=init_range,
                                     quantity=quantity,
                                     neg_list=list(max_deg_node_sample))
    middle_deg_node_sample = get_sample(dict_deg_to_node_list=dict_deg_to_node_list,
                                        max_deg=max_deg, min_deg=min_deg,
                                        sampling_node_deg_center=middle_deg, sampling_node_deg_radius=init_range,
                                        quantity=quantity,
                                        neg_list=list(max_deg_node_sample) + list(min_deg_node_sample))

    samples = max_deg_node_sample + middle_deg_node_sample + min_deg_node_sample

    return samples


def sample_without_graph_splitting_nodes(graph: gc.Graph, quantity: int) -> List[int]:
    """
    Returns a list of nodes in the graph which does not split the graph into two or more components.
    :param graph: graph the nodes should be slected from
    :param quantity: number of nodes to sample
    :return: list of non splitting nodes from the graph
    """
    nodes_list_without_spitting = __filter_splitting_nodes(node_list=graph.nodes(), graph=graph)
    return list(np.random.choice(a=nodes_list_without_spitting, size=quantity, replace=False))


def __list_index_of_at_least_value(list: List[int], value: int, max_value: int):
    # try larger values
    while value <= max_value:
        try:
            return list.index(value)
        except ValueError:
            value = value + 1
    return -1


def __list_index_of_at_most_value(list: List[int], value: int, min_value: int):
    # try smaller values
    while value >= 1:
        try:
            return list.index(value)
        except ValueError:
            value = value - 1
    return -1


def __remove_entities_from_list(input_list: List, entries_to_remove: List):
    return list(filter(lambda entry: entry not in entries_to_remove, input_list))


def get_sample(dict_deg_to_node_list: Dict[int, List[int]], min_deg: int, max_deg: int,
               sampling_node_deg_center: int, sampling_node_deg_radius: int, quantity: int,
               neg_list: List[int]) -> List[int]:
    """
    This function generates a sample of 'quantity' nodes in the sorted_dict_nodes_degrees randomly among nodes in range
    (sampling_node_deg_center-sampling_node_deg_radius,sampling_node_deg_center+sampling_node_deg_radius).
    If there are not enough nodes in this range, alls thoses nodes are sampled and the range is successively increased
    :param dict_deg_to_node_list: dictionary nodes -> degree, sorted by degree.
    :param sampling_node_deg_center: center of degree range nodes are sampled from
    :param sampling_node_deg_radius: defines the radius (or offset) around the center of the degree range nodes are
    sampled from
    :param min_deg: minimal node degree
    :param max_deg: maximal node degree
    :param quantity: number of nodes to sample
    :param neg_list: nodes which should not be sampled
    :return: 
    """

    sampled_nodes = []
    candidates = []

    lower_degree_bound = sampling_node_deg_center - sampling_node_deg_radius
    upper_degree_bound = sampling_node_deg_center + sampling_node_deg_radius

    for deg in range(lower_degree_bound, upper_degree_bound + 1):
        candidates = candidates + dict_deg_to_node_list.get(deg, [])
        candidates = __remove_entities_from_list(input_list=candidates, entries_to_remove=neg_list)

    node_radius_increase = 0
    while len(candidates) + len(sampled_nodes) < quantity \
            and (lower_degree_bound > min_deg or upper_degree_bound < max_deg):
        sampled_nodes += candidates
        node_radius_increase += 1
        candidates = dict_deg_to_node_list.get(
            sampling_node_deg_center - sampling_node_deg_radius - node_radius_increase, [])
        candidates += dict_deg_to_node_list.get(
            sampling_node_deg_center + sampling_node_deg_radius + node_radius_increase, [])
        candidates = __remove_entities_from_list(input_list=candidates, entries_to_remove=neg_list)

    if len(sampled_nodes) + len(candidates) < quantity:
        raise ValueError(
            f"__get_sample: Could not sample {quantity} nodes. Not enough nodes are available."
            f"Degree node dict: {dict_deg_to_node_list}, neg_list: {neg_list}."
        )

    sampled_nodes += list(np.random.choice(a=candidates, size=quantity - len(sampled_nodes), replace=False))

    return sampled_nodes


def __filter_splitting_nodes(node_list: List[int], graph: gc.Graph):
    return list(filter(lambda x: not graph.splits_graph(x), node_list))


def __get_filtered_random_nodes(all_other_list: List[int], num_needed_nodes: int, graph: gc.Graph):
    candidates = np.random.choice(a=all_other_list, size=num_needed_nodes, replace=False)

    target_list = __filter_splitting_nodes(node_list=candidates, graph=graph)
    num_needed_nodes = num_needed_nodes - len(target_list)

    if num_needed_nodes > 0:
        all_other_list = list(filter(lambda elem: elem not in candidates, all_other_list))
        target_list += __get_filtered_random_nodes(all_other_list=all_other_list, num_needed_nodes=num_needed_nodes,
                                                   graph=graph)

    return target_list
