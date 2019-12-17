import os
import subprocess
from typing import List

import config
import entities.graphs.graph as g
import memory_access.memory_access as ma
import entities.embeddings.embedding as emb
import gensim


class Line(emb.Embedding):

    def __init__(self, dim: int = 128, threshold: int = 1000, depth: int = 2):
        assert (dim % 2 == 0)  # will be devided by 2
        self.dim: int = dim  # 2 embeddings will be created and added together each emb has size dim
        self.threshold = threshold
        self.depth = depth

    def __str__(self):
        return f'LINE-dim={self.dim}_depth={self.depth}_threshold={self.threshold}'

    def short_name(self) -> str:
        return "LINE"

    def train_embedding(self, graph: g.Graph, memory_access: ma.MemoryAccess, removed_nodes: [int]):
        super().train_embedding(graph=graph, memory_access=memory_access, removed_nodes=removed_nodes)

        if memory_access.has_embedding(emb_func_name=str(self), graph_name=str(graph), removed_nodes=removed_nodes):
            # embedding is already trained
            return

        dense_edge_list = self.__get_preprocessed_edge_list(removed_nodes=removed_nodes, graph=graph,
                                                            mem_acc=memory_access)

        target_emb_path_file = self.__train_embedding(dense_edge_list_file_path=dense_edge_list,
                                                      mem_acc=memory_access, removed_nodes=removed_nodes, graph=graph)
        os.remove(dense_edge_list)

        self.__change_embedding_emb_format_to_csv(target_emb_path_file=target_emb_path_file, graph=graph,
                                                  removed_nodes=removed_nodes, mem_acc=memory_access)

    def __change_embedding_emb_format_to_csv(self, target_emb_path_file: str, graph: g.Graph, removed_nodes: List[int],
                                             mem_acc: ma.MemoryAccess):
        assert (target_emb_path_file.endswith(".emb"))
        model = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(target_emb_path_file, binary=True)
        mem_acc.save_gensim_embedding(trained_emb=model, emb_func_name=str(self), graph_name=str(graph),
                                      removed_nodes=removed_nodes, graph_nodes=graph.nodes())
        os.remove(target_emb_path_file)

    @staticmethod
    def __get_preprocessed_edge_list(graph: g.Graph, mem_acc: ma.MemoryAccess, removed_nodes: [int]) -> str:
        edge_list_file_path = mem_acc.access_edge_list_file_path(graph_name=str(graph), removed_nodes=removed_nodes,
                                                                 edge_list=graph.edges())

        path_name = os.path.splitext(edge_list_file_path)[0]
        directed_weighted_edge_list = path_name + ".directedWeightedEdgelist"
        dense_edge_list = path_name + ".denseEdgelist"

        if os.path.exists(dense_edge_list):
            print("dense edge list already exists")
            return dense_edge_list

        if not os.path.exists(edge_list_file_path):
            raise ValueError(f"Edge list does not exist: {edge_list_file_path}")

        working_dir = os.getcwd()
        os.chdir(config.LINE_DIR)
        subprocess.call(f'python preprocess_youtube.py "{edge_list_file_path}" "{directed_weighted_edge_list}"',
                        shell=True)

        if not os.path.exists(directed_weighted_edge_list):
            raise ValueError(f"Directed weighted edge list could not be computed. Target file: {edge_list_file_path}")

        subprocess.call(
            f'./reconstruct -train "{directed_weighted_edge_list}" -output "{dense_edge_list}" '
            f'-depth 2 -threshold 1000',
            shell=True)
        os.chdir(working_dir)
        os.remove(directed_weighted_edge_list)

        if not os.path.exists(dense_edge_list):
            raise ValueError(f"Dense edge list could not be computed. Target file {dense_edge_list}")

        return dense_edge_list

    def __train_embedding(self, dense_edge_list_file_path: str, graph: g.Graph, mem_acc: ma.MemoryAccess,
                          removed_nodes: List[int]) -> str:
        target_file_path = mem_acc.get_embedding_path_name(emb_func_name=str(self), graph_name=str(graph),
                                                           removed_nodes=removed_nodes)

        target_emb_path_file = target_file_path + ".emb"

        first_order_emb = target_file_path + "_order_1.emb"
        second_order_emb = target_file_path + "_order_2.emb"
        norm_first_order_emb = target_file_path + "_order_1_normalised.emb"
        norm_second_order_emb = target_file_path + "_order_2_normalised.emb"

        # execute embedding
        working_dir = os.getcwd()
        os.chdir(config.LINE_DIR)
        assert (os.path.exists(dense_edge_list_file_path))

        subprocess.call(
            f'./line -train "{dense_edge_list_file_path}" -output "{first_order_emb}" -size \
                {str(self.dim / 2)} -order 1 -binary 1 -threads {config.NUM_CORES}',
            shell=True)
        subprocess.call(
            f'./line -train "{dense_edge_list_file_path}" -output "{second_order_emb}" -size \
                {str(self.dim / 2)} -order 2 -binary 1 -threads {config.NUM_CORES}',
            shell=True)
        subprocess.call(f'./normalize -input "{first_order_emb}" -output "{norm_first_order_emb}" -binary 1',
                        shell=True)
        subprocess.call(f'./normalize -input "{second_order_emb}" -output "{norm_second_order_emb}" -binary 1',
                        shell=True)
        subprocess.call(
            f'./concatenate -input1 "{norm_first_order_emb}" -input2 "{norm_second_order_emb}" '
            f'-output "{target_emb_path_file}" -binary 1',
            shell=True)
        os.chdir(working_dir)

        assert (os.path.exists(target_emb_path_file))

        # remove unnecessary files to save memory
        os.remove(first_order_emb)
        os.remove(second_order_emb)
        os.remove(norm_first_order_emb)
        os.remove(norm_second_order_emb)

        return target_emb_path_file
