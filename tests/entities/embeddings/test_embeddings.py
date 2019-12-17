from unittest import TestCase

import main.entities.embeddings.gem_embedding as gem
import main.entities.embeddings.line as line
import main.entities.embeddings.node2vec as n2v
import main.entities.graphs.graph as g

import memory_access.memory_access_test_extension as ma


class TestEmbedding(TestCase):
    def setUp(self) -> None:
        self.graph = g.Graph.init_karate_club_graph()
        self.graph._name = "test_graph"
        self.mem_acc = ma.MemoryAccessTestExtension()

    def tearDown(self) -> None:
        # self.mem_acc.delete_directory(graph_name=str(self.graph))
        pass

    def test_local_linear_embedding(self):
        dim = 8
        emb_method = gem.GEMEmbedding.init_local_linear_embedding(dim=dim)
        emb_method.train_embedding(graph=self.graph, memory_access=self.mem_acc, removed_nodes=[])
        trained_emb = self.mem_acc.load_embedding(emb_func_name=str(emb_method), graph_name=str(self.graph),
                                                  removed_nodes=[])
        self.assertEqual(trained_emb.shape, (len(self.graph.nodes()), dim))

    def test_init_graph_factorisation(self):
        dim = 8
        emb_method = gem.GEMEmbedding.init_graph_factorisation(dim=dim)
        emb_method.train_embedding(graph=self.graph, memory_access=self.mem_acc, removed_nodes=[])
        trained_emb = self.mem_acc.load_embedding(emb_func_name=str(emb_method), graph_name=str(self.graph),
                                                  removed_nodes=[])
        self.assertEqual(trained_emb.shape, (len(self.graph.nodes()), dim))

    def test_hope(self):
        dim = 8
        emb_method = gem.GEMEmbedding.init_hope(dim=dim)
        emb_method.train_embedding(graph=self.graph, memory_access=self.mem_acc, removed_nodes=[])
        trained_emb = self.mem_acc.load_embedding(emb_func_name=str(emb_method), graph_name=str(self.graph),
                                                  removed_nodes=[])
        self.assertEqual(trained_emb.shape, (len(self.graph.nodes()), dim))

    def test_sdne(self):
        dim = 128
        emb_method = gem.GEMEmbedding.init_sdne(dim=dim)
        emb_method.train_embedding(graph=self.graph, memory_access=self.mem_acc, removed_nodes=[])
        trained_emb = self.mem_acc.load_embedding(emb_func_name=str(emb_method), graph_name=str(self.graph),
                                                  removed_nodes=[])
        self.assertEqual(trained_emb.shape, (len(self.graph.nodes()), dim))

    def test_line(self):
        dim = 8
        emb_method = line.Line(dim=dim)
        emb_method.train_embedding(graph=self.graph, memory_access=self.mem_acc, removed_nodes=[])
        trained_emb = self.mem_acc.load_embedding(emb_func_name=str(emb_method), graph_name=str(self.graph),
                                                  removed_nodes=[])
        self.assertEqual(trained_emb.shape, (len(self.graph.nodes()), dim))

    def test_n2v(self):
        dim = 8
        emb_method = n2v.Node2Vec(dim=8)
        emb_method.train_embedding(graph=self.graph, memory_access=self.mem_acc, removed_nodes=[])
        trained_emb = self.mem_acc.load_embedding(emb_func_name=str(emb_method), graph_name=str(self.graph),
                                                  removed_nodes=[])
        self.assertEqual(trained_emb.shape, (len(self.graph.nodes()), dim))
