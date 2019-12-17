from unittest import TestCase, expectedFailure
from parameterized import parameterized
from typing import Dict, List, Tuple, Set

import ast

import main.logic.node_sampling.sampler as sam
import main.entities.graphs.graph as gc


class TestSampling(TestCase):

    @parameterized.expand([
        ["normal", "5", "1", "3", "{5:[0],4:[1],7:[2,3],2:[4,5],8:[6,7,8]}", "[]", "[0,1]", "[2,3]"],
        ["exact nodes with deg in center", "5", "1", "3", "{5:[0,1,2],3:[3],2:[4,5],8:[6,7,8]}", "[]", "[0,1,2]", "[]"],
        ["more nodes with deg in center", "5", "1", "3", "{5:[0,1,2,3],3:[9],2:[4,5],8:[6,7,8]}", "[]", "[]",
         "[0,1,2,3]"],
        ["exact nodes with deg in range", "5", "1", "3", "{5:[0],4:[1],6:[2],2:[3,4,5],8:[6,7,8]}", "[]", "[0,1,2]",
         "[]"],
        ["contains nodes with deg in range", "5", "1", "3", "{5:[0],4:[1],6:[2,3],2:[4,5],8:[6,7,8]}", "[]", "[]",
         "[0,1,2,3]"],
        ["exact nodes with deg in range + 1", "5", "1", "3", "{5:[0],4:[1],7:[2],2:[3,4,5],8:[6,7,8]}", "[]",
         "[0,1,2]", "[]"],
        ["no node in range", "5", "1", "3", "{7:[2,3],2:[0,1,4],0:[5],9:[6,7,8]}", "[]", "[2,3]", "[0,1,4]"],
        ["find min degree", "1", "1", "3", "{1:[0],2:[1],3:[2,3],4:[4,5],8:[6,7,8]}", "[]", "[0,1]", "[2,3]"],
        ["find max degree", "10", "1", "3", "{10:[0],9:[1],8:[2,3],2:[4,5],7:[6,7,8]}", "[]", "[0,1]", "[2,3]"],
        ["test neg list in range", "5", "1", "3", "{5:[0],4:[1],7:[2,3],2:[4,5],8:[6,7,8]}", "[1]", "[0,2,3]", "[]"],
        ["test neg list in range + 1 ", "5", "1", "3", "{5:[0],4:[1],7:[2,3],2:[4,5],8:[6,7,8]}", "[2]", "[0,1,3]",
         "[]"],
    ])
    def test_get_sample(self, name: str, deg_center: int, deg_radius: int, quantity: int,
                        dict_deg_to_node_list: Dict[int, int], neg_list: List[int],
                        must_target: List[int], may_target: List[int]):
        with self.subTest(name):
            # given
            quantity = int(quantity)
            deg_center = int(deg_center)
            deg_radius = int(deg_radius)
            dict_deg_to_node_list = ast.literal_eval(dict_deg_to_node_list)
            neg_list = ast.literal_eval(neg_list)
            min_deg = min(dict_deg_to_node_list.keys())
            max_deg = max(dict_deg_to_node_list.keys())
            # when
            result = sam.get_sample(dict_deg_to_node_list=dict_deg_to_node_list, min_deg=min_deg,
                                    max_deg=max_deg, quantity=quantity,
                                    sampling_node_deg_center=deg_center, sampling_node_deg_radius=deg_radius,
                                    neg_list=neg_list)

            # then
            must_target = set(ast.literal_eval(must_target))
            may_target = set(ast.literal_eval(may_target))

            self.assertEqual(len(result), quantity, msg=f"Result did not sample enough nodes. Result: {result}")
            self.assertEqual(len(set(result)), len(result),
                             msg=f"Results contains duplicates! Result {result}")  # no duplicates

            self.assertTrue(set(result).issuperset(must_target), msg=f"Nodes with must be sampled are not in result "
                                                                     f"Result: {result}, Target: {must_target}")
            self.assertTrue(set(result).issubset(must_target.union(may_target)),
                            msg=f"Sampled nodes are not in the target Nodes. "
                                f"Result: {result}, Must Target: {must_target}, May Target: {may_target}")

    @expectedFailure
    def test_get_sample_not_enough_nodes(self):
        # given
        quantity = 3
        deg_center = 5
        deg_radius = 1
        dict_deg_node_list = {5: [0], 4: [1]}
        neg_list = []
        min_deg = min(dict_deg_node_list.keys())
        max_deg = max(dict_deg_node_list.keys())

        # when/then
        sam.get_sample(dict_deg_to_node_list=dict_deg_node_list, min_deg=min_deg, max_deg=max_deg,
                       quantity=quantity,
                       sampling_node_deg_center=deg_center, sampling_node_deg_radius=deg_radius,
                       neg_list=neg_list)

    def __set_intersect_contains_exact_num_of_values(self, set1: Set[int], set2: Set[int],
                                                     num_of_intersecting_values: int) -> bool:
        return len(set1.intersection(set2)) == num_of_intersecting_values

    @parameterized.expand([
        ["normal", "1", "0", "[]", "[0,1]", "[5,6]", "[4,9]"],
        ["exact", "1", "0", "[]", "[0,1]", "[5,6]", "[4,9]"],
        ["no node with deg", "1", "0", "[[5,6]]", "[0,1]", "[5,6,7,8,9,11]", "[4,9]"]
    ])
    def test_sample_low_avg_high_degree_nodes(self, name: str, quantity: int, init_range: int,
                                              add_edges: List[Tuple[int, int]],
                                              low_deg_target: List[int], middle_deg_target: List[int],
                                              high_deg_target: List[int]):
        with self.subTest(name):
            # given
            quantity = int(quantity)
            init_range = int(init_range)
            edge_list = [[0, 2], [1, 3], [2, 3], [2, 4], [2, 5], [3, 4], [3, 6], [4, 5], [4, 6], [4, 8], [4, 9],
                         [5, 7], [5, 8], [6, 9], [6, 10], [7, 8], [7, 11], [8, 9], [8, 11], [9, 10], [9, 11]]
            edge_list += ast.literal_eval(add_edges)
            graph = gc.Graph(name="test_graph", nodes=list(range(12)), edges=edge_list)

            # when
            result = sam.sample_low_avg_high_degree_nodes(graph=graph, quantity=quantity, init_range=init_range)

            # then
            low_deg_target = set(ast.literal_eval(low_deg_target))
            middle_deg_target = set(ast.literal_eval(middle_deg_target))
            high_deg_target = set(ast.literal_eval(high_deg_target))

            self.assertEqual(len(result), quantity * 3, msg=f"Result did not sample the right amount of nodes."
                                                            f" Result: {result}")
            self.assertEqual(len(set(result)), len(result), msg=f"Results contains duplicates! Result {result}")

            self.assertTrue(self.__set_intersect_contains_exact_num_of_values(low_deg_target, set(result), quantity),
                            msg=f" Result does not contain enough low deg nodes Result {result}, "
                                f"Low deg candidates: {low_deg_target}")
            self.assertTrue(self.__set_intersect_contains_exact_num_of_values(middle_deg_target, set(result), quantity),
                            msg=f" Result does not contain enough middle deg nodes. "
                                f"Should contain {quantity} nodes. Result {result}, "
                                f"middle deg candidates: {middle_deg_target}")
            self.assertTrue(self.__set_intersect_contains_exact_num_of_values(high_deg_target, set(result), quantity),
                            msg=f" Result does not contain enough high deg nodes Result {result}, "
                                f"middle deg candidates: {high_deg_target}")

    def test_sample_low_avg_high_degree_nodes_test_all_nodes_same_deg(self):
        # given
        quantity = 1
        init_range = 0
        edge_list = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        graph = gc.Graph(name="test_graph", nodes=list(range(4)), edges=edge_list)

        # when
        result = sam.sample_low_avg_high_degree_nodes(graph=graph, quantity=quantity, init_range=init_range)

        # then
        self.assertEqual(len(result), quantity * 3, msg=f"Result did not sample the right amount of nodes."
                                                        f" Result: {result}")
        self.assertEqual(len(set(result)), len(result), msg=f"Results contains duplicates! Result {result}")
        self.assertTrue(set(result).issubset(set(graph.nodes())))

    @parameterized.expand([
        ["exact number of nodes available", "4", "[0,2,3,5]"],
        ["more nodes available", "1", "[0,2,3,5]"]
    ])
    def test_sample_without_graph_splitting_nodes(self, name: str, quantity, target: List[int]):
        # given
        edge_list = [(0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (4, 5)]
        graph = gc.Graph("test", nodes=list(range(6)), edges=edge_list)
        quantity = int(quantity)

        # when
        result = sam.sample_without_graph_splitting_nodes(graph=graph, quantity=quantity)

        # then
        target = ast.literal_eval(target)
        self.assertEqual(len(result), len(set(result)), msg=f"A nodes is sampled multiple times. Results {result}")
        self.assertEqual(len(result), quantity, msg="The number of sampled nodes is not equal to quantity."
                                                    f"Result {result}, quantity {quantity}")
        self.assertTrue(set(result).issubset(target),
                        msg=f"Not all sampled nodes are in the target nodes. Result {result}, target {target}")

    def test_sample_without_graph_splitting_nodes_not_enough_nodes(self):
        # given
        edge_list = [(0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (4, 5)]
        graph = gc.Graph("test", nodes=list(range(6)), edges=edge_list)
        quantity = 5

        # when
        with self.assertRaises(ValueError):
            sam.sample_without_graph_splitting_nodes(graph=graph, quantity=quantity)
