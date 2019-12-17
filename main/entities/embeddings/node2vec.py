import subprocess
from typing import List

import os
import config
import gensim
import os.path

import entities.embeddings.embedding as e
import entities.graphs.graph as g

import memory_access.memory_access as ma


class Node2Vec(e.Embedding):

    def __init__(self, dim: int = 128, epochs: object = 5, window_size: int = 10, walk_length: int = 80,
                 num_of_walks_per_node: int = 10, alpha: float = 0.025):
        self.dim: int = dim
        self.epochs: int = epochs
        self.window_size: int = window_size
        self.walk_length: int = walk_length
        self.num_of_walks_per_node: int = num_of_walks_per_node
        self.alpha: float = alpha

    def __str__(self):
        return f'Node2Vec_dim={self.dim}' \
               f'_epochs={self.epochs}' \
               f'_windowSize={self.window_size}' \
               f'_walkLength={self.walk_length}' \
               f'_walksPerNode={self.num_of_walks_per_node}' \
               f'_p=1_q=1_alpha_{self.alpha}'

    def short_name(self):
        return "Node2Vec"

    def train_embedding(self, graph: g.Graph, memory_access: ma.MemoryAccess,
                        removed_nodes: List[int]):
        super().train_embedding(graph=graph, memory_access=memory_access, removed_nodes=removed_nodes)

        edge_list_path = memory_access.access_edge_list_file_path(graph_name=str(graph), removed_nodes=removed_nodes,
                                                                  edge_list=graph.edges())

        self.train_node2vec_embedding(edge_list_path=edge_list_path, graph=graph,
                                      removed_nodes=removed_nodes, mem_acc=memory_access)

    def train_node2vec_embedding(self, edge_list_path: str,
                                 graph: g.Graph,
                                 removed_nodes: [int], mem_acc: ma.MemoryAccess):

        target = mem_acc.get_embedding_path_name(emb_func_name=str(self), graph_name=str(graph),
                                                 removed_nodes=removed_nodes)

        target_path = os.path.abspath(target + "_path.emb")

        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        # create walks

        # execute path training
        working_dir = os.getcwd()
        os.chdir(config.NODE2VEC_SNAP_DIR)

        emb_call = f'./node2vec  -i:"{edge_list_path}" -o:"{target_path}"  -e:{str(self.epochs)} ' \
                   f'-d:{self.dim} -l:{self.walk_length} -r:{self.num_of_walks_per_node} -k:{self.window_size} -ow'
        subprocess.call(emb_call, shell=True)

        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Random walks for computing node2vec embedding for "
                                    f"removed nodes {removed_nodes} and network {str(graph)} was not successful. "
                                    f"Make sure that snap is correctly and is accessible from the command list!")
        os.chdir(working_dir)

        # end create paths

        class Walks:
            def __init__(self, file):
                self.file = file

            def __iter__(self):
                with open(target_path, "r") as f:
                    for line in f:
                        line = line.strip("\n").split(" ")
                        # assert (all(list(map(lambda node: node in graph.nodes(), list(map(int, line))))))
                        yield line

        walks = Walks(target_path)

        # train word2vec
        emb_result = gensim.models.Word2Vec(walks, size=self.dim, iter=self.epochs, window=self.window_size,
                                            min_count=1, sg=1,
                                            workers=config.NUM_CORES, alpha=self.alpha)

        os.remove(target_path)

        mem_acc.save_gensim_embedding(trained_emb=emb_result, emb_func_name=str(self), graph_name=str(graph),
                                      removed_nodes=removed_nodes, graph_nodes=graph.nodes())
