from unittest import TestCase
import pandas as pd
import numpy as np

import main.entities.bins as b
import main.entities.graphs.graph as g
import main.logic.compute_features.compute_features as cf


class TestComputeFeatures(TestCase):
    def test_compute_cosine_distance_matrix(self):
        # given
        embedding = pd.DataFrame([[0.5, 0.5, 0.5, 0.5], [0.9, 0.3, 0.3, 0.1]], index=[0, 1], columns=[0, 1, 2, 4])

        # when
        cos_matrix: pd.DataFrame = cf.compute_cosine_distance_matrix(embedding)

        # then
        np.testing.assert_array_equal(cos_matrix.index.values, embedding.index.values)
        np.testing.assert_array_equal(cos_matrix.columns.values, embedding.index.values)
        pd.testing.assert_frame_equal(cos_matrix, pd.DataFrame(([0, 0.2], [0.2, 0])))

    def test_compute_difference_matrix_by_difference(self):
        # given
        dist_matrix = pd.DataFrame([[0, 2, 3, 4], [2, 0, 6, 7], [3, 6, 0, 8], [4, 7, 8, 0]], dtype=float)

        # when
        diff_matrix = cf.compute_difference_matrix_by_difference(distance_matrix_one=dist_matrix,
                                                                 distance_matrix_two=dist_matrix)

        # then
        pd.testing.assert_frame_equal(diff_matrix, pd.DataFrame(np.zeros((4, 4))))

    def test_compute_bins(self):
        # given
        diff_matrix = pd.DataFrame([[0, 2, 3, 4], [2, 0, 6, 7], [3, 6, 0, 8], [4, 7, 8, 0]])

        # when
        bins = cf.compute_bins(diff_matrix, num_of_bins=2)

        # then
        np.testing.assert_array_equal(bins.bin_boundaries, [2, 5, 8])

    def test_compute_bins_not_symmetric(self):
        # given
        diff_matrix = pd.DataFrame([[0, 2, 3, 4], [0, 0, 6, 7], [3, 6, 0, 8], [4, 7, 8, 0]])

        # when
        with self.assertRaises(AssertionError):
            cf.compute_bins(diff_matrix, num_of_bins=2)

    def test_compute_compute_num_dist_values_in_bin_per_node(self):
        # given
        diff_matrix = pd.DataFrame([[0, 2, 3, 4], [2, 0, 6, 7], [3, 6, 0, 8], [4, 7, 8, 0]])
        bins = b.Bins(bin_boundaries=np.array([2, 5, 8]))

        # when
        features = cf.compute_num_dist_values_in_bin_per_node(diff_matrix=diff_matrix, bins=bins)

        # then
        pd.testing.assert_frame_equal(features, pd.DataFrame([[3, 0], [1, 2], [1, 2], [1, 2]], dtype=float))

    def test_compute_compute_num_dist_values_in_bin_per_node_across_0(self):
        # given
        diff_matrix = pd.DataFrame([[0, -0.3, 0.5, -0.2], [-0.3, 0, 1.0, 0], [0.5, 1.0, 0, -0.7], [-0.2, 0, -0.7, 0]])
        bins = b.Bins(bin_boundaries=np.array([-0.7, -0.3, 0.1, 0.5, 1.0]))

        # when
        result = cf.compute_num_dist_values_in_bin_per_node(diff_matrix=diff_matrix, bins=bins)

        # then
        pd.testing.assert_frame_equal(result, pd.DataFrame([[1, 2, 1, 0], [1, 2, 0, 1], [1, 1, 1, 1], [1, 3, 0, 0]],
                                                           dtype=float))

    def test_compute_compute_num_dist_values_values_below_bins(self):
        # given
        diff_matrix = pd.DataFrame([[0, -0.8, 0.5, -0.2], [-0.3, 0, 1.0, 0], [0.5, 1.0, 0, -0.7], [-0.2, 0, -0.7, 0]])
        bins = b.Bins(bin_boundaries=np.array([-0.7, -0.3, 0.1, 0.5, 1.0]))

        # when
        with self.assertRaises(AssertionError):
            result = cf.compute_num_dist_values_in_bin_per_node(diff_matrix=diff_matrix, bins=bins)

    def test_compute_compute_num_dist_values_values_above_bins(self):
        # given
        diff_matrix = pd.DataFrame([[0, 1.1, 0.5, -0.2], [-0.3, 0, 1.0, 0], [0.5, 1.0, 0, -0.7], [-0.2, 0, -0.7, 0]])
        bins = b.Bins(bin_boundaries=np.array([-0.7, -0.3, 0.1, 0.5, 1.0]))

        # when
        with self.assertRaises(AssertionError):
            result = cf.compute_num_dist_values_in_bin_per_node(diff_matrix=diff_matrix, bins=bins)

    def test_compute_compute_num_dist_values_matrix_not_symmetric(self):
        # given
        diff_matrix = pd.DataFrame([[0, -0.1, 0.5, -0.2], [-0.3, 0, 1.0, 0], [0.5, 1.0, 0, -0.7], [-0.2, 0, -0.7, 0]])
        bins = b.Bins(bin_boundaries=np.array([-0.7, -0.3, 0.1, 0.5, 1.0]))

        # when
        with self.assertRaises(AssertionError):
            result = cf.compute_num_dist_values_in_bin_per_node(diff_matrix=diff_matrix, bins=bins)

    def test_compute_degree_column(self):
        # given
        graph = g.Graph.init_karate_club_graph()
        node_deg_dict = {2: 10, 5: 4, 11: 1, 31: 6}

        # when
        deg_col = cf.compute_degree_column(node_list=list(node_deg_dict.keys()), graph=graph)

        # then
        target_column = pd.DataFrame([[node_deg_dict[2]], [node_deg_dict[5]], [node_deg_dict[11]], [node_deg_dict[31]]],
                                     index=list(node_deg_dict.keys()), columns=["deg"])

        pd.testing.assert_frame_equal(deg_col, target_column)
