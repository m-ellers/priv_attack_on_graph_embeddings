import sys
import config
import pandas as pd

import entities.graphs.graph as g
import entities.embeddings.embedding as emb
import memory_access.memory_access as ma

sys.path.insert(0, config.GEM_EMBEDDING_DIR)
import gem.embedding.static_graph_embedding as abs_emb
import gem.embedding.lle as lle
import gem.embedding.gf as gf
import gem.embedding.hope as hope
import gem.embedding.sdne as sdne


class GEMEmbedding(emb.Embedding):

    def __init__(self, embedding: abs_emb, is_static: bool = False):
        self.__gem_embedding: abs_emb = embedding
        self.__is_static = is_static

    def __str__(self):
        return f"{self.__gem_embedding.get_method_summary()}"

    def short_name(self):
        return f"{self.__gem_embedding.get_method_name()}"

    def train_embedding(self, graph: g.Graph, memory_access: ma.MemoryAccess,
                        removed_nodes: [int]):
        super().train_embedding(graph=graph, memory_access=memory_access, removed_nodes=removed_nodes)

        if memory_access.has_embedding(emb_func_name=str(self), graph_name=str(graph), removed_nodes=removed_nodes):
            # embedding is already trained
            return

        nx_g = graph.to_networkx()
        nx_g.to_directed()
        # nx_g = nx.convert_node_labels_to_integers(nx_g) # should already be the case

        y, t = self.__gem_embedding.learn_embedding(graph=nx_g, is_weighted=False, no_python=True)

        trained_embedding = pd.DataFrame(y, index=graph.nodes())

        memory_access.save_embedding(trained_emb=trained_embedding, emb_func_name=str(self), graph_name=str(graph),
                                     removed_nodes=removed_nodes)

    @staticmethod
    def init_local_linear_embedding(dim: int = 128):
        return GEMEmbedding(lle.LocallyLinearEmbedding(d=dim), is_static=True)

    @staticmethod
    def init_graph_factorisation(dim: int = 128, max_iter: int = 1000, eta: float = 1 * 10 ** -4, regu: float = 1.0):
        return GEMEmbedding(gf.GraphFactorization(d=dim, max_iter=max_iter, eta=eta, regu=regu), is_static=True)

    @staticmethod
    def init_hope(dim: int = 128, beta: float = 0.01):
        return GEMEmbedding(hope.HOPE(d=dim, beta=beta))

    @staticmethod
    def init_sdne(dim: int = 128, beta: int = 5, alpha: float = 1e-5, nu1: float = 1e-6, nu2: float = 1e-6, k: int = 3,
                  n_units=None, rho: float = 0.3, n_iter: int = 30, xeta: float = 0.001,
                  n_batch: int = 500):
        if n_units is None:
            n_units = [500, 300, ]
        return GEMEmbedding(
            sdne.SDNE(d=dim, beta=beta, alpha=alpha, nu1=nu1, nu2=nu2, K=k, n_units=n_units, rho=rho, n_iter=n_iter,
                      xeta=xeta, n_batch=n_batch))
