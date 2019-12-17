from unittest import TestCase

import main.entities.graphs.graph as g


class TestGraph(TestCase):

    def test_init_DBLP_graph_moderate_homophily_snowball_sampled_2000(self):
        graph = g.Graph.init_dblp_graph_moderate_homophily_snowball_sampled_2000()
        self.assertIsNotNone(graph)
        self.assertEqual(2000, len(graph.nodes()))
        self.assertEqual(7036, len(graph.edges()))
        self.assertEqual('DBLP_graph_moderate_homophily_snowball_sampled_2000', graph.name())

    def test_init_barabasi_graph_m2(self):
        graph = g.Graph.init_barabasi_m2_n1000()
        self.assertIsNotNone(graph)
        self.assertEqual(1000, len(graph.nodes()))
        self.assertEqual(1996, len(graph.edges()))
        self.assertEqual('barabasi_m2_n1000', graph.name())

    def test_init_barabasi_graph_m5(self):
        graph = g.Graph.init_barabasi_m5_n1000()
        self.assertIsNotNone(graph)
        self.assertEqual(1000, len(graph.nodes()))
        self.assertEqual(4975, len(graph.edges()))
        self.assertEqual('barabasi_m5_n1000', graph.name())

    def test_init_barabasi_graph_m10(self):
        graph = g.Graph.init_barabasi_m10_n1000()
        self.assertIsNotNone(graph)
        self.assertEqual(1000, len(graph.nodes()))
        self.assertEqual(9900, len(graph.edges()))
        self.assertEqual('barabasi_m10_n1000', graph.name())

    def test_init_barabasi_graph_m20(self):
        graph = g.Graph.init_barabasi_m20_n1000()
        self.assertIsNotNone(graph)
        self.assertEqual(1000, len(graph.nodes()))
        self.assertEqual(19600, len(graph.edges()))
        self.assertEqual('barabasi_m20_n1000', graph.name())

    def test_init_barabasi_graph_m50(self):
        graph = g.Graph.init_barabasi_m50_n1000()
        self.assertIsNotNone(graph)
        self.assertEqual(1000, len(graph.nodes()))
        self.assertEqual(47500, len(graph.edges()))
        self.assertEqual('barabasi_m50_n1000', graph.name())

    def test_init_barabasi_graph_m5_n100(self):
        graph = g.Graph.init_barabasi_m5_n100()
        self.assertIsNotNone(graph)
        self.assertEqual(100, len(graph.nodes()))
        self.assertEqual(475, len(graph.edges()))
        self.assertEqual('barabasi_m5_n100', graph.name())

    def test_init_barabasi_graph_m5_n500(self):
        graph = g.Graph.init_barabasi_m5_n500()
        self.assertIsNotNone(graph)
        self.assertEqual(500, len(graph.nodes()))
        self.assertEqual(2475, len(graph.edges()))
        self.assertEqual('barabasi_m5_n500', graph.name())

    def test_init_barabasi_graph_m5_n2000(self):
        graph = g.Graph.init_barabasi_m5_n2000()
        self.assertIsNotNone(graph)
        self.assertEqual(2000, len(graph.nodes()))
        self.assertEqual(9975, len(graph.edges()))
        self.assertEqual('barabasi_m5_n2000', graph.name())

    def test_init_barabasi_graph_m5_n5000(self):
        graph = g.Graph.init_barabasi_m5_n5000()
        self.assertIsNotNone(graph)
        self.assertEqual(5000, len(graph.nodes()))
        self.assertEqual(24975, len(graph.edges()))
        self.assertEqual('barabasi_m5_n5000', graph.name())

    def test_init_barabasi_graph_m5_n10000(self):
        graph = g.Graph.init_barabasi_m5_n10000()
        self.assertIsNotNone(graph)
        self.assertEqual(10000, len(graph.nodes()))
        self.assertEqual(49975, len(graph.edges()))
        self.assertEqual('barabasi_m5_n10000', graph.name())

    def test_init_hamsterster_cc(self):
        graph = g.Graph.init_hamsterster_cc()
        self.assertIsNotNone(graph)
        self.assertEqual(1788, len(graph.nodes()))
        self.assertEqual(12476, len(graph.edges()))
        self.assertEqual(f'hamsterster_cc', graph.name())

    def test_avg_neighbour_degree_same_degree_same_degree(self):
        # given
        graph = g.Graph.init_karate_club_graph()

        # when
        avg_n_deg = graph.average_neighbour_degree(node=16)

        # then
        self.assertEqual(4, avg_n_deg)

    def test_avg_neighbour_degree_one_neighbour(self):
        # given
        graph = g.Graph.init_karate_club_graph()

        # when
        avg_n_deg = graph.average_neighbour_degree(node=11)

        # then
        self.assertEqual(16, avg_n_deg)

    def test_avg_neighbour_degree(self):
        # given
        graph = g.Graph.init_karate_club_graph()

        # when
        avg_n_deg = graph.average_neighbour_degree(node=17)

        # then
        self.assertEqual(12.5, avg_n_deg)

    def test_distance_dist1(self):
        # given
        graph = g.Graph.init_karate_club_graph()

        # when
        dist = graph.distance(0, 1)

        # then
        self.assertEqual(1, dist)

    def test_distance_dist0(self):
        # given
        graph = g.Graph.init_karate_club_graph()

        # when
        dist = graph.distance(0, 0)

        # then
        self.assertEqual(0, dist)

    def test_distance_dist_larger(self):
        # given
        graph = g.Graph.init_karate_club_graph()

        # when
        dist = graph.distance(17, 13)

        # then
        self.assertEqual(2, dist)

    def test_two_hop_neighbours(self):
        # given
        graph = g.Graph.init_karate_club_graph()
        # when
        two_hop_neighbours = graph.two_hop_neighbours(node=7)

        # then
        target = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 19, 21, 27, 28, 30, 31, 32}
        self.assertSetEqual(target, set(two_hop_neighbours))

    def test_splits_graph(self):
        edges = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (3, 5)]
        graph = g.Graph.init_from_list_of_edges(name="test", edges=edges)

        self.assertTrue(graph.splits_graph(node=2))
        self.assertTrue(graph.splits_graph(node=2))
        self.assertFalse(graph.splits_graph(node=0))
        self.assertTrue(graph.splits_graph(node=3))
        self.assertFalse(graph.splits_graph(node=5))

    def test_remove_self_loops(self):
        # given
        edges = [(0, 1), (0, 2), (1, 2), (2, 3), (0, 0), (2, 2)]
        graph = g.Graph.init_from_list_of_edges(name="test", edges=edges)

        # when
        graph.remove_self_loops()

        # then
        self.assertSetEqual(set(graph.edges()), {(0, 1), (0, 2), (1, 2), (2, 3)})

    def test_add_node(self):
        # given
        nodes = [0, 1, 2, 3]
        edges = [(0, 1), (0, 2), (1, 2), (2, 3), (0, 0), (2, 2)]
        graph = g.Graph.init_from_list_of_edges(name="test", edges=edges)

        neighbours_of_new_node = [0, 2]
        add_name = "test"
        # when
        new_graph = graph.add_node(neighbours=neighbours_of_new_node, name_addition=add_name)

        # then
        self.assertEqual(nodes + [14], new_graph.nodes())
        self.assertSetEqual(set(edges + [(0, 14), (2, 14)]), set(new_graph.edges()))
        self.assertEqual(f'{str(graph)}_add_node_neighbours_{add_name}', str(new_graph))

    def test_density(self):
        nodes = [0, 1, 2, 3]
        edges = [(0, 1), (0, 2), (1, 2), (2, 3), (0, 0), (2, 2)]
        graph = g.Graph.init_from_list_of_edges(name="test", edges=edges)

        self.assertEqual(1, graph.density())

    def test_density_2(self):
        nodes = [0, 1, 2, 3]
        edges = [(0, 1), (0, 2), (1, 2), (2, 3)]
        graph = g.Graph.init_from_list_of_edges(name="test", edges=edges)

        self.assertEqual(2 / 3, graph.density())

    def test_triangle_count_one(self):
        nodes = [0, 1, 2, 3]
        edges = [(0, 1), (0, 2), (1, 2), (2, 3)]
        graph = g.Graph.init_from_list_of_edges(name="test", edges=edges)

        triangle_count = graph.triangle_count()

        self.assertEqual(1, triangle_count)

    def test_triangle_count_many(self):
        nodes = [0, 1, 2, 3]
        edges = [(0, 1), (0, 2), (1, 2), (2, 3), (0, 3)]
        graph = g.Graph.init_from_list_of_edges(name="test", edges=edges)

        triangle_count = graph.triangle_count()

        self.assertEqual(2, triangle_count)

    def test_triangle_count_zero(self):
        nodes = [0, 1, 2, 3]
        edges = [(0, 1), (1, 2), (2, 3)]
        graph = g.Graph.init_from_list_of_edges(name="test", edges=edges)

        triangle_count = graph.triangle_count()

        self.assertEqual(0, triangle_count)
