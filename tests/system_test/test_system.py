import functools
from typing import List
from unittest import TestCase
import sklearn.naive_bayes as nb


import main.entities.parameters as p
import main.entities.graphs.graph as g
import main.entities.embeddings.gem_embedding as gem
import main.memory_access.memory_access_test_extension as ma
import main.pipeline as pipeline


class TestSystem(TestCase):
    def setUp(self) -> None:
        self.mem_acc = ma.MemoryAccessTestExtension()
        self.graph_name = "test_graph_system_test"
        self.emb_name = "test_emb"

    def tearDown(self) -> None:
        self.mem_acc.delete_directory(graph_name=self.graph_name)

    def test_pipeline(self):
        # given
        num_attacked_nodes_per_deg_level = 5
        num_training_graphs = 5

        num_of_bins = 10
        embedding = gem.GEMEmbedding.init_hope(dim=8)
        classifier = nb.GaussianNB()

        graph = g.Graph.init_karate_club_graph()
        graph._name = self.graph_name  # rename to use different dir for testing

        # when
        dict_attacked_nodes_training_nodes = pipeline.sample_nodes(
            graph=graph,
            num_attacked_nodes_pre_deg_level=num_attacked_nodes_per_deg_level,
            num_training_nodes=num_training_graphs)

        params = p.Parameters(graph=graph,
                              embedding=embedding,
                              num_of_bins=num_of_bins,
                              classifier=classifier,
                              dict_attacked_nodes_training_nodes=dict_attacked_nodes_training_nodes)

        pipeline.run(parameters=params, memory_access=self.mem_acc)

        # then
        list_attacked_nodes = [[node] for node in list(params.dict_attacked_nodes_training_nodes.keys())]
        list_pair_attacked_trained_node = list(functools.reduce(
            lambda l, a_node: l + [[a_node, tr_node] for tr_node in
                                   params.dict_attacked_nodes_training_nodes[a_node]],
            list(params.dict_attacked_nodes_training_nodes.keys()), []))

        self.__assert_all_embeddings_trained(params=params, list_attacked_nodes=list_attacked_nodes,
                                             list_pair_attacked_trained_node=list_pair_attacked_trained_node)
        self.__assert_all_features_trained(params=params, list_attacked_nodes=list_attacked_nodes,
                                           list_pair_attacked_trained_node=list_pair_attacked_trained_node)
        self.__assert_all_models_trained(params=params)
        self.__assert_test_data_per_node_saved(params=params)
        self.__assert_agg_test_data_saved(params=params)

    def __assert_all_embeddings_trained(self, params: p.Parameters, list_attacked_nodes: List[List[int]],
                                        list_pair_attacked_trained_node: List[List[int]]):
        for rem_nodes in [] + list_attacked_nodes + list_pair_attacked_trained_node:
            with self.subTest(f"test if embedding is trained: removed nodes {rem_nodes}"):
                assert (
                    self.mem_acc.has_embedding(emb_func_name=str(params.embedding), graph_name=str(params.graph),
                                               removed_nodes=rem_nodes))

    def __assert_all_features_trained(self, params: p.Parameters, list_attacked_nodes: List[List[int]],
                                      list_pair_attacked_trained_node: List[List[int]]):
        for rem_nodes in list_attacked_nodes + list_pair_attacked_trained_node:
            with self.subTest(f"test if features are trained: removed nodes {rem_nodes}"):
                self.assertTrue(
                    self.mem_acc.has_embedding(emb_func_name=str(params.embedding), graph_name=str(params.graph),
                                               removed_nodes=rem_nodes))

    def __assert_all_models_trained(self, params: p.Parameters):
        for attacked_node in list(params.dict_attacked_nodes_training_nodes.keys()):
            with self.subTest(f"test if model is trained: attacked node {attacked_node} "
                              f"tr nodes: {params.dict_attacked_nodes_training_nodes[attacked_node]}"):
                self.assertTrue(
                    self.mem_acc.has_classification_model(
                        classifier_name=str(params.classifier), emb_func_name=str(params.embedding),
                        graph_name=str(params.graph), attacked_node=attacked_node,
                        training_nodes=params.dict_attacked_nodes_training_nodes[attacked_node],
                        num_bins=params.num_of_bins)
                )

    def __assert_test_data_per_node_saved(self, params: p.Parameters):
        with self.subTest("test test-data per node"):
            self.assertTrue(
                self.mem_acc.has_test_results_per_node(
                    dict_attack_train_nodes=params.dict_attacked_nodes_training_nodes,
                    emb_func_name=str(params.embedding), graph_name=str(params.graph),
                    classifier_name=str(params.classifier), num_bins=params.num_of_bins)
            )

    def __assert_agg_test_data_saved(self, params: p.Parameters):
        with self.subTest("test aggregated test results"):
            self.assertTrue(
                self.mem_acc.has_aggregated_test_results(
                    dict_attack_train_nodes=params.dict_attacked_nodes_training_nodes,
                    emb_func_name=str(params.embedding), graph_name=str(params.graph),
                    classifier_name=str(params.classifier), num_bins=params.num_of_bins)
            )
