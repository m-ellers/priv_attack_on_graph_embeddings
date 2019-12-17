from unittest import TestCase

import pandas as pd

import main.entities.graphs.graph as g
import main.logic.classify.compute_labels as gl


class TestComputeLabels(TestCase):
    def test_compute_labels(self):
        # given
        graph = g.Graph.init_karate_club_graph()
        removed_node = 2
        node_neighbor_dict = {0: True, 5: False, 7: True, 8: True, 10: False, 31: False, 32: True, 33: False}

        # when
        computed_labels = gl.compute_labels(node_list=list(node_neighbor_dict.keys()), graph=graph,
                                            removed_node=removed_node)

        # then
        target_labels = pd.Series(list(node_neighbor_dict.values()), index=list(node_neighbor_dict.keys()))
        pd.testing.assert_series_equal(computed_labels, target_labels)
