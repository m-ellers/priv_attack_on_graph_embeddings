import os
from typing import List
from unittest import TestCase
from parameterized import parameterized

import pandas as pd
import numpy as np
import ast

import sklearn.naive_bayes as nb

import main.entities.graphs.graph as g
import main.memory_access.memory_access_test_extension as ma


class TestMemoryAccess(TestCase):
    def setUp(self) -> None:
        self.mem_acc = ma.MemoryAccessTestExtension()
        self.graph_name = "test_graph"
        self.emb_name = "test_emb"

    def tearDown(self) -> None:
        self.mem_acc.delete_directory(graph_name=self.graph_name)

    def test_embedding_access(self):
        # given
        trained_emb = pd.DataFrame(np.random.rand(3, 3))

        # when
        self.mem_acc.save_embedding(trained_emb=trained_emb, emb_func_name=self.emb_name, graph_name=self.graph_name,
                                    removed_nodes=[])

        # then
        has_emb = self.mem_acc.has_embedding(emb_func_name=self.emb_name, graph_name=self.graph_name, removed_nodes=[])
        self.assertTrue(has_emb)

        loaded_embedding = self.mem_acc.load_embedding(emb_func_name=self.emb_name, graph_name=self.graph_name,
                                                       removed_nodes=[])

        self.assertEqual((3, 3), loaded_embedding.shape)
        self.assertTrue(trained_emb.equals(loaded_embedding))

    def test_access_edge_list_file_path(self):
        # given
        edge_list = [(0, 1), (1, 2), (2, 3)]

        # when
        edge_list_file_path = self.mem_acc.access_edge_list_file_path(graph_name=self.graph_name, removed_nodes=[],
                                                                      edge_list=edge_list)

        graph = g.Graph.init_from_edge_list(edge_list_file_path)

        # then
        self.assertTrue(os.path.exists(edge_list_file_path), msg="Edge list was not created!")
        self.assertEqual(len(graph.edges()), len(edge_list), msg="not all edges are loaded")
        self.assertSetEqual(set(edge_list), set(graph.edges()), msg="input edge list differs from output edge list")

    @parameterized.expand(["[]", "[42]", "[0,3]"])
    def test_features_access(self, rem_nodes: List[int]):
        # given
        num_bins = 3
        features = pd.DataFrame(np.random.rand(7, num_bins))
        rem_nodes = ast.literal_eval(rem_nodes)
        # when
        self.mem_acc.save_features(features=features, emb_func_name=self.emb_name, graph_name=self.graph_name,
                                   removed_nodes=rem_nodes, num_bins=num_bins)
        has_emb = self.mem_acc.has_features(emb_func_name=self.emb_name, graph_name=self.graph_name,
                                            removed_nodes=rem_nodes, num_bins=num_bins)

        loaded_features = self.mem_acc.load_features(emb_func_name=self.emb_name, graph_name=self.graph_name,
                                                     removed_nodes=rem_nodes, num_bins=num_bins)

        # then
        self.assertTrue(has_emb)
        self.assertEqual(features.shape, loaded_features.shape)

    def test_training_feature_access(self):
        # given
        num_bins = 4
        num_nodes = 7
        tr_features_list = [pd.DataFrame(np.random.rand(num_nodes, num_bins)),
                            pd.DataFrame(np.random.rand(num_nodes, num_bins)),
                            pd.DataFrame(np.random.rand(num_nodes, num_bins))]
        attacked_node = 42
        # when
        for rem_second_node, tr_features in enumerate(tr_features_list):
            self.mem_acc.save_features(features=tr_features, emb_func_name=self.emb_name,
                                       graph_name=self.graph_name, removed_nodes=[attacked_node, rem_second_node],
                                       num_bins=num_bins)

        loaded_features = self.mem_acc.load_training_features(emb_func_name=self.emb_name, graph_name=self.graph_name,
                                                              attacked_node=attacked_node,
                                                              training_nodes=list(range(len(tr_features_list))),
                                                              num_bins=num_bins)

        # then
        self.assertEqual(loaded_features.shape, (3 * num_nodes, num_bins))
        pd.testing.assert_frame_equal(loaded_features,
                                      tr_features_list[0].append(tr_features_list[1]).append(tr_features_list[2]))

    def test_model_access(self):
        # given
        num_bins = 3
        attacked_node = 42
        training_nodes = [1, 2, 3]
        model = nb.GaussianNB()
        model.fit(X=[[0.9, 0.0, 0.1], [0.1, 0.9, 0.0]], y=[True, False])

        # when
        self.mem_acc.save_classification_model(classification_model=model, emb_func_name=self.emb_name,
                                               graph_name=self.graph_name,
                                               attacked_node=attacked_node, training_nodes=training_nodes,
                                               num_bins=num_bins)

        has_model = self.mem_acc.has_classification_model(classifier_name=str(model), emb_func_name=self.emb_name,
                                                          graph_name=self.graph_name,
                                                          attacked_node=attacked_node, training_nodes=training_nodes,
                                                          num_bins=num_bins)

        loaded_model = self.mem_acc.load_classification_model(classifier_name=str(model), emb_func_name=self.emb_name,
                                                              graph_name=self.graph_name,
                                                              attacked_node=attacked_node,
                                                              training_nodes=training_nodes,
                                                              num_bins=num_bins)

        # then
        self.assertTrue(has_model)
        self.assertEqual(str(loaded_model), str(model))
        self.assertTrue(loaded_model.predict([[0.8, 0.1, 0.1]])[0])
        self.assertFalse(loaded_model.predict([[0.1, 0.9, 0.1]])[0])

    def test_access_test_results_per_node(self):
        # given
        num_bins = 3
        dict_attack_train_nodes = {0: [1, 2], 42: [3, 4], 52: [0, 1]}
        classifier_name = "Classifier"
        test_results = pd.DataFrame([[0, 0], [0.5, 0.5], [0.7, 0.7]], index=[0, 42, 52])

        # when
        self.mem_acc.save_test_results_per_node(results_per_node=test_results,
                                                dict_attack_train_nodes=dict_attack_train_nodes,
                                                emb_func_name=self.emb_name, graph_name=self.graph_name,
                                                num_bins=num_bins, classifier_name=classifier_name)

        has_test_results = self.mem_acc.has_test_results_per_node(dict_attack_train_nodes=dict_attack_train_nodes,
                                                                  emb_func_name=self.emb_name,
                                                                  graph_name=self.graph_name,
                                                                  num_bins=num_bins, classifier_name=classifier_name)

        loaded_test_results = self.mem_acc.load_test_results_per_node(dict_attack_train_nodes=dict_attack_train_nodes,
                                                                      emb_func_name=self.emb_name,
                                                                      graph_name=self.graph_name,
                                                                      num_bins=num_bins,
                                                                      classifier_name=classifier_name)

        # then
        self.assertTrue(has_test_results)
        pd.testing.assert_frame_equal(loaded_test_results, test_results)

    def test_access_agg_test_results(self):
        # given
        num_bins = 3
        dict_attack_train_nodes = {0: [1, 2], 42: [3, 4], 52: [0, 1]}
        classifier_name = "Classifier"
        test_results = pd.DataFrame([[0, 0], [0.5, 0.5], [0.7, 0.7]], index=[0, 42, 52], columns=[0, 2])

        # when
        self.mem_acc.save_aggregated_test_results(results_per_node=test_results,
                                                  dict_attack_train_nodes=dict_attack_train_nodes,
                                                  emb_func_name=self.emb_name, graph_name=self.graph_name,
                                                  num_bins=num_bins, classifier_name=classifier_name)

        has_test_results = self.mem_acc.has_aggregated_test_results(dict_attack_train_nodes=dict_attack_train_nodes,
                                                                    emb_func_name=self.emb_name,
                                                                    graph_name=self.graph_name,
                                                                    num_bins=num_bins, classifier_name=classifier_name)

        loaded_test_results = self.mem_acc.load_aggregated_test_results(dict_attack_train_nodes=dict_attack_train_nodes,
                                                                        emb_func_name=self.emb_name,
                                                                        graph_name=self.graph_name,
                                                                        num_bins=num_bins,
                                                                        classifier_name=classifier_name)

        # then
        self.assertTrue(has_test_results)
        pd.testing.assert_frame_equal(loaded_test_results, test_results)
