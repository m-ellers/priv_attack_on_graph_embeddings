import networkx as nx
import config
from typing import List, Tuple, Iterable
import functools


class Graph:
    def __init__(self, name: str, nodes: List[int], edges: List[Tuple[int, int]]):
        self._name: str = name
        self._nodes: list = list(sorted(nodes))
        self._edges: list = edges

    @classmethod
    def init_from_list_of_edges(cls, edges: List[Tuple[int, int]], name: str):
        nodes = set()
        for edge in edges:
            nodes.add(edge[0])
            nodes.add(edge[1])
        return cls(name=name, nodes=sorted(list(nodes)), edges=list(edges))

    @classmethod
    def init_from_networkx(cls, g: nx.Graph):
        name = str(g)
        g = nx.convert_node_labels_to_integers(g)
        return cls(name, list(g.nodes()), list(g.edges()))

    @classmethod
    def init_from_gexf(cls, path: str):
        if not path.endswith(".gexf"):
            raise ValueError("Path does not end with .gexf")

        g = nx.read_gexf(path)
        g.name = path.split("/")[-1][:-5]  # remove file suffix
        return cls.init_from_networkx(g)

    @classmethod
    def __get_homophily_gefx_link(cls, graph_name: str):
        return cls.__graph_base_dir() + f"{graph_name}.gexf"

    @classmethod
    def __get_snowball_sampled_homophily(cls, graph_name: str):
        return cls.__graph_base_dir() + f"{graph_name}.edgelist"

    @classmethod
    def __get_subsampled_homophily_gefx_link(cls, graph_name: str):
        return cls.__graph_base_dir() + f"subsampled_homophily/{graph_name}.gexf"

    @classmethod
    def __get_gen_graph_link(cls):
        return cls.__graph_base_dir() + f"generated-graphs/"

    @classmethod
    def __init_barabasi_graph(cls, n, m):
        return cls.init_from_edge_list(Graph.__get_gen_graph_link() + f"barabasi_m{m}_n{n}.edgelist")

    @staticmethod
    def __graph_base_dir():
        return f"{config.PROJECT_PATH}data/"

    @classmethod
    def init_from_edge_list(cls, edge_list_file: str, name: str = None):
        edges: List[Tuple[int, int]] = []
        nodes = set()
        with open(edge_list_file, "r") as f:
            # assert (edge_list_file.endswith(".edgelist") or edge_list_file.endswith(".edges"))
            lines = f.readlines()
            for line in lines:
                if line.strip().startswith("%"):
                    continue
                line = line.replace('\t', ' ')
                edge = tuple(map(int, (line.strip('\n').split(" ")[:2])))
                edges.append(edge)
        for edge in edges:
            nodes.add(edge[0])
            nodes.add(edge[1])
        if name is None:
            name = edge_list_file.split("/")[-1].strip(".edgelist")
        return cls(name=name, nodes=list(nodes), edges=edges)

    """
    Available Graphs init
    """

    @classmethod
    def init_karate_club_graph(cls) -> 'Graph':
        return cls.init_from_networkx(nx.karate_club_graph())

    @classmethod
    def init_dblp_graph_moderate_homophily(cls):
        return cls.init_from_gexf(Graph.__get_homophily_gefx_link("DBLP_graph_moderate_homophily"))

    @classmethod
    def init_github_mutual_follower_ntw(cls):
        return cls.init_from_gexf(Graph.__get_homophily_gefx_link("github_mutual_follower_ntw"))

    @classmethod
    def init_pok_max_cut_high_heterophily(cls):
        return cls.init_from_gexf(Graph.__get_homophily_gefx_link("pok_max_cut"))

    @classmethod
    def __get_facebook_link(cls):
        return cls.__graph_base_dir() + "facebook/"

    @classmethod
    def init_list_of_homophily_graphs(cls):
        yield cls.init_pok_max_cut_high_heterophily()
        yield cls.init_github_mutual_follower_ntw()
        yield cls.init_dblp_graph_moderate_homophily()

    @classmethod
    def init_dblp_graph_moderate_homophily_snowball_sampled_2000(cls):
        return cls.init_from_edge_list(
            cls.__get_snowball_sampled_homophily("DBLP_graph_moderate_homophily_snowball_sampled_2000"))

    @classmethod
    def init_list_of_snowball_sampled_2000_homophily_graphs(cls):
        yield cls.init_dblp_graph_moderate_homophily_snowball_sampled_2000()
        yield cls.init_facebook_wosn_2009_snowball_sampled_2000()

    @classmethod
    def init_barabasi_m2_n1000(cls):
        return cls.__init_barabasi_graph(n=1000, m=2)

    @classmethod
    def init_barabasi_m5_n1000(cls):
        return cls.__init_barabasi_graph(n=1000, m=5)

    @classmethod
    def init_barabasi_m10_n1000(cls):
        return cls.__init_barabasi_graph(n=1000, m=10)

    @classmethod
    def init_barabasi_m20_n1000(cls):
        return cls.__init_barabasi_graph(n=1000, m=20)

    @classmethod
    def init_barabasi_m50_n1000(cls):
        return cls.__init_barabasi_graph(n=1000, m=50)

    @classmethod
    def init_barabasi_m5_n100(cls):
        return cls.__init_barabasi_graph(n=100, m=5)

    @classmethod
    def init_barabasi_m5_n500(cls):
        return cls.__init_barabasi_graph(n=500, m=5)

    @classmethod
    def init_barabasi_m5_n2000(cls):
        return cls.__init_barabasi_graph(n=2000, m=5)

    @classmethod
    def init_barabasi_m5_n5000(cls):
        return cls.__init_barabasi_graph(n=5000, m=5)

    @classmethod
    def init_barabasi_m5_n10000(cls):
        return cls.__init_barabasi_graph(n=10000, m=5)

    @classmethod
    def init_list_of_barabasi_graphs_with_different_density(cls):
        yield cls.init_barabasi_m2_n1000()
        yield cls.init_barabasi_m5_n1000()
        yield cls.init_barabasi_m10_n1000()
        yield cls.init_barabasi_m20_n1000()
        yield cls.init_barabasi_m50_n1000()

    @classmethod
    def init_list_of_barabasi_graphs_with_different_size(cls):
        yield cls.init_barabasi_m5_n100()
        yield cls.init_barabasi_m5_n500()
        yield cls.init_barabasi_m5_n1000()
        yield cls.init_barabasi_m5_n2000()
        yield cls.init_barabasi_m5_n5000()
        # yield Graph.init_barabasi_m5_n10000()

    @classmethod
    def init_all_but_barabasi(cls):
        yield cls.init_facebook_wosn_2009_snowball_sampled_2000()
        yield cls.init_hamsterster_cc()
        yield cls.init_dblp_graph_moderate_homophily_snowball_sampled_2000()

    @classmethod
    def init_all_different_graphs(cls):
        yield cls.init_barabasi_m5_n1000()
        yield from cls.init_all_but_barabasi()

    @classmethod
    def init_all_barabasi_graphs(cls):
        yield from cls.init_list_of_barabasi_graphs_with_different_density()
        yield from cls.init_list_of_barabasi_graphs_with_different_size()

    @classmethod
    def init_list_of_all_used_graphs(cls):
        yield from cls.init_all_but_barabasi()
        yield from cls.init_list_of_barabasi_graphs_with_different_density()
        yield from cls.init_list_of_barabasi_graphs_with_different_size()

    @classmethod
    def init_hamsterster_cc(cls) -> "Graph":
        return cls.init_from_edge_list(Graph.__graph_base_dir() + "hamsterster_cc.edgelist",
                                       name="hamsterster_cc")

    @classmethod
    def init_facebook_wosn_2009_snowball_sampled_2000(cls) -> "Graph":
        return cls.init_from_edge_list(Graph.__graph_base_dir() + "facebook_wosn_snowball_sampled_2000.edgelist",
                                       name="facebook_wosn_snowball_sampled_2000")

    @classmethod
    def init_email_eu_core_cc(cls) -> "Graph":
        """
            Source : https://snap.stanford.edu/data/email-Eu-core.html
        """
        return cls.init_from_edge_list(Graph.__graph_base_dir() + "email-Eu-core_cc.edgelist",
                                       name="email_eu_core_cc")

    def edges(self):
        return self._edges

    def nodes(self):
        return self._nodes

    def name(self):
        return self._name

    def __str__(self):
        return self._name

    @functools.lru_cache(maxsize=None, typed=False)
    def neighbours(self, node: int):
        assert (node in self.nodes())
        neighbours: list = []
        for edge in self._edges:
            if node in edge:
                neighbours.append(edge[0] if edge[1] == node else edge[1])
        return neighbours

    def two_hop_neighbours(self, node: int):
        neighbours: list = self.neighbours(node)
        two_hop_neighbours = set(neighbours)
        for neighbour in neighbours:
            two_hop_neighbours = two_hop_neighbours.union(set(self.neighbours(neighbour)))
        return list(two_hop_neighbours)

    def delete_node(self, removed_node: int) -> 'Graph':
        new_nodes = self.nodes().copy()
        try:
            new_nodes.remove(removed_node)
        except ValueError:
            raise ValueError("Node {} is not in the graph, hence can not be removed!".format(removed_node))

        new_edges = list(filter(lambda edge: edge[0] != removed_node and edge[1] != removed_node, self.edges()))

        return Graph(name=self._name, nodes=new_nodes, edges=new_edges)

    def add_fully_connected_node(self, node_name: int, inplace: bool = False):
        if node_name in self.nodes():
            raise ValueError("Node {} is already in the Graph. Nodes:{}".format(node_name, self.nodes()))

        new_edges = [(node_name, node) for node in self.nodes()]

        if inplace:
            self._edges.extend(new_edges)
            self._nodes.append(node_name)
        else:
            e = self.edges().copy()
            n = self.nodes().copy()
            e.extend(new_edges)
            n.append(node_name)
            return Graph(self._name, n, e)
        # self._edges.append()

    def copy(self) -> 'Graph':
        n = self.nodes().copy()
        e = self.edges().copy()
        return Graph(name=self.name(), nodes=n, edges=e)

    def to_networkx(self) -> nx.Graph:
        nx_g = nx.Graph()
        nx_g.add_nodes_from(self.nodes())
        nx_g.add_edges_from(self.edges())
        return nx_g

    def degree(self, node: int):
        return len(self.neighbours(node))

    def all_degrees(self):
        return list(map(lambda node: self.degree(node), self.nodes()))

    def get_neighbour_dict(self):
        d = dict()
        for node in self.nodes():
            d[node] = []

        for edge in self.edges():
            d[edge[0]].append(edge[1])
            d[edge[1]].append(edge[0])
        return d

    def average_neighbour_degree(self, node):
        neighbours = self.neighbours(node)
        deg_sum = 0
        for neighbour in self.neighbours(node):
            deg_sum += self.degree(neighbour)

        return deg_sum / len(neighbours)

    def distance(self, node1, node2):
        """
        Waring: Very Inefficient
        :param node1:
        :param node2:
        :return: distance between node1 and node2
        """
        try:
            dist = nx.shortest_path_length(self.to_networkx(), source=node1, target=node2)
        except Exception:
            dist = 9999
        return dist

    def betweenness_centrality(self):
        return nx.betweenness_centrality(self.to_networkx())

    def remove_self_loops(self):
        self._edges = list(filter(lambda e: e[0] != e[1], self.edges()))

    def is_connected(self):
        return nx.is_connected(self.to_networkx())

    @functools.lru_cache(maxsize=32)
    def splits_graph(self, node: int):
        """
        tests if the graph is split by removing node "node"
        :param node: the node that might split the graph
        :return: bool if node splits graph. True if it splits the graph
        :return: bool if node splits graph. True if it splits the graph
        """
        gnx = self.to_networkx()
        gnx.remove_node(n=node)

        return not nx.is_connected(gnx)

    def add_node(self, neighbours: Iterable[int], name_addition: str) -> "Graph":
        new_node_name = len(self.nodes()) + 10
        new_edges = list(map(lambda n: (n, new_node_name), neighbours))

        return Graph(name=self.name() + f'_add_node_neighbours_{name_addition}', nodes=self.nodes() + [new_node_name],
                     edges=self.edges() + new_edges)

    def density(self):
        return nx.density(self.to_networkx())

    def triangle_count(self):
        return sum(nx.triangles(self.to_networkx()).values()) / 3
