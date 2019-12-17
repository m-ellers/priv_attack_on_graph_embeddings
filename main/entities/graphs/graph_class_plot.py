from typing import List, Tuple

import entities.graphs.graph as g
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

sns.set()


class GraphPlot(g.Graph):

    def __init__(self, name: str, nodes: List[int], edges: List[Tuple[int, int]]):
        super().__init__(name=name, nodes=nodes, edges=edges)

    @staticmethod
    def init_from_graph(graph: g.Graph):
        graph = GraphPlot(nodes=graph.nodes(), edges=graph.edges(), name=graph.name())
        graph.draw()

    def draw(self):
        nx_g = self.to_networkx()
        plt.title(str(self))
        nx.draw(nx_g, pos=nx.spring_layout(nx_g), node_size=2, with_labels=True)
        plt.show()

    def plot_deg_distribution(self):
        degs = self.all_degrees()
        sns.distplot(degs)
        plt.show()


if __name__ == '__main__':
    g = GraphPlot.init_karate_club_graph()
    print(type(g))
