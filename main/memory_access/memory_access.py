import os
from typing import List, Tuple, Dict
import sklearn.base as sk_base
import pandas as pd
import config
import joblib
import gensim


class MemoryAccess:
    def __init__(self, base_dir: str = None):
        """
        Data Management class for loading and writing to/from the memory
        :param base_dir: path to dir in which trained data is saved. Default: config.DIR_PATH
        """
        if base_dir is None:
            self.base_dir = config.DIR_PATH
        else:
            self.base_dir = base_dir
        print('Base directory path used:', self.base_dir)

    # --- private methods ---

    def _get_graph_base_path(self, graph_name: str) -> str:
        return self.base_dir + f"priv_attack_data/graph_name-{graph_name}/"

    def _get_base_path(self, graph_name: str, embedding_name: str) -> str:
        return self._get_graph_base_path(graph_name=graph_name) + f"embedding_type-{embedding_name}/"

    def _get_embedding_path(self, graph_name: str, embedding_name: str) -> str:
        return self._get_base_path(graph_name=graph_name, embedding_name=embedding_name) + "embeddings/"

    def _get_feature_path(self, graph_name: str, emb_func_name: str) -> str:
        return self._get_base_path(graph_name=graph_name, embedding_name=emb_func_name) + "features/"

    def _get_graphs_path(self, graph_name: str):
        return self._get_graph_base_path(graph_name=graph_name) + "graphs/"

    def _get_model_path(self, graph_name, emb_func_name):
        return self._get_base_path(graph_name=graph_name, embedding_name=emb_func_name) + "models/"

    def _get_test_results_per_node_path(self, graph_name, emb_func_name):
        return self._get_base_path(graph_name=graph_name,
                                   embedding_name=emb_func_name) + "test_results_per_attacked_node/"

    def _get_agg_test_results_path(self, graph_name, emb_func_name):
        return self._get_base_path(graph_name=graph_name,
                                   embedding_name=emb_func_name) + "test_results_aggregated/"

    def _get_graph_path_name(self, graph_name: str, removed_nodes: List[int]) -> str:
        return self._get_graphs_path(graph_name=graph_name) + f"edge_list_removed_nodes_{str(sorted(removed_nodes))}"

    def get_embedding_path_name(self, emb_func_name: str, graph_name: str, removed_nodes: List[int]):
        return self._get_embedding_path(graph_name=graph_name, embedding_name=emb_func_name) + \
               f"emb_removed_nodes_{str(sorted(removed_nodes))}"

    def _get_feature_path_name(self, emb_func_name: str, graph_name: str, removed_nodes: List[int], num_bins: int):
        return self._get_feature_path(graph_name=graph_name, emb_func_name=emb_func_name) + \
               f"features_removed_nodes_{str(removed_nodes)}_num_bins_{num_bins}"

    def _get_model_path_name(self, emb_func_name: str, graph_name: str, num_bins: int,
                             classifier_name: str, attacked_node: int, training_nodes: List[int]):
        return self._get_model_path(graph_name=graph_name, emb_func_name=emb_func_name) + \
               f"model_nodes_{str(attacked_node)}_tr_nodes_{str(sorted(training_nodes))}" \
               f"_num_bins_{num_bins}_classifier_{classifier_name}"

    @staticmethod
    def __compress_dict_in_key_and_count(dictionary: Dict[int, List[int]]) -> str:
        # compress information to reduce filename length
        output: str = ""
        for key in dictionary:
            output += f"{key}n{len(dictionary[key])},"
        return output

    def _get_test_results_per_attacked_node_path_name(self, emb_func_name: str, graph_name: str, num_bins: int,
                                                      classifier_name: str,
                                                      dict_attack_train_nodes: Dict[int, List[int]]):

        return self._get_test_results_per_node_path(graph_name=graph_name, emb_func_name=emb_func_name) + \
               f"test_results_attacked_nodes_{self.__compress_dict_in_key_and_count(dict_attack_train_nodes)}_" \
               f"num_bins_{num_bins}_classifier_{classifier_name}"

    def _get_agg_test_results_path_name(self, emb_func_name: str, graph_name: str, num_bins: int,
                                        classifier_name: str,
                                        dict_attack_train_nodes: Dict[int, List[int]]):
        return self._get_agg_test_results_path(graph_name=graph_name, emb_func_name=emb_func_name) + \
               f"agg_test_results_nodes_{self.__compress_dict_in_key_and_count(dict_attack_train_nodes)}_" \
               f"num_bins_{num_bins}_classifier_{classifier_name}"

    # --- static helper methods ---

    @staticmethod
    def __dict_to_path_str(dictionary: Dict):
        return str(dictionary).replace(' ', '').replace(':', '_')

    @staticmethod
    def __save_csv_compressed(file_name: str, csv: pd.DataFrame) -> None:
        MemoryAccess.__assure_path_exists(os.path.dirname(file_name))
        try:
            csv.to_pickle(file_name + ".csv.gz", compression="gzip")
        except Exception as e:
            # remove file on interrupt to prevent corrupted files
            if os.path.exists(file_name + ".csv.gz"):
                os.remove(file_name + ".csv.gz")
            raise e

    @staticmethod
    def __load_compressed_csv(file_name: str) -> pd.DataFrame:
        try:
            csv: pd.DataFrame = pd.read_pickle(file_name + ".csv.gz", compression="gzip")
        except Exception as e:
            import sys
            raise type(e)(
                f"{str(e)}\n Could not load file with filename: '{file_name}.csv.gz'\n {e}").with_traceback(
                sys.exc_info()[2])
        return csv

    @staticmethod
    def __assure_path_exists(path: str) -> str:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        return path

    # --- public methods ---

    @staticmethod
    def __has_compressed_csv(file_name: str) -> bool:
        return os.path.exists(file_name + ".csv.gz")

    @staticmethod
    def __compute_df_from_gensim_model(trained_emb: gensim.models.word2vec.Word2Vec, nodes: List[int]):
        df_embedding = pd.DataFrame(columns=list(range(trained_emb.vector_size)), index=nodes)
        for node in nodes:
            df_embedding.loc[node] = trained_emb.wv.get_vector(str(node))
        return df_embedding

    def save_gensim_embedding(self, trained_emb: gensim.models.word2vec.Word2Vec, emb_func_name: str, graph_name: str,
                              removed_nodes: List[int], graph_nodes: List[int]) -> None:
        self.save_embedding(trained_emb=self.__compute_df_from_gensim_model(trained_emb=trained_emb, nodes=graph_nodes),
                            emb_func_name=emb_func_name,
                            graph_name=graph_name, removed_nodes=removed_nodes)

    def save_embedding(self, trained_emb: pd.DataFrame, emb_func_name: str, graph_name: str,
                       removed_nodes: List[int]) -> None:
        self.__save_csv_compressed(
            file_name=self.get_embedding_path_name(emb_func_name=emb_func_name, graph_name=graph_name,
                                                   removed_nodes=removed_nodes),
            csv=trained_emb)

    def load_embedding(self, emb_func_name: str, graph_name: str, removed_nodes: List[int]) -> pd.DataFrame:
        return self.__load_compressed_csv(
            file_name=self.get_embedding_path_name(emb_func_name=emb_func_name, graph_name=graph_name,
                                                   removed_nodes=removed_nodes))

    def has_embedding(self, emb_func_name: str, graph_name: str, removed_nodes: List[int]) -> bool:
        return self.__has_compressed_csv(
            file_name=self.get_embedding_path_name(emb_func_name=emb_func_name, graph_name=graph_name,
                                                   removed_nodes=removed_nodes))

    def access_edge_list_file_path(self, graph_name: str, removed_nodes: List[int], edge_list: List[Tuple[int, int]]):
        file_name = self._get_graph_path_name(graph_name=graph_name, removed_nodes=removed_nodes) + ".edgelist"
        if not os.path.exists(file_name):
            MemoryAccess.__assure_path_exists(os.path.dirname(file_name))
            edges = "\n".join(list(map(lambda edge: str(edge[0]) + " " + str(edge[1]), edge_list)))
            with open(file_name, "w+") as file:
                file.write(edges)
        return file_name

    def has_features(self, emb_func_name: str, graph_name: str, removed_nodes: List[int], num_bins: int) -> bool:
        file_name = self._get_feature_path_name(graph_name=graph_name, emb_func_name=emb_func_name,
                                                removed_nodes=removed_nodes, num_bins=num_bins)
        return self.__has_compressed_csv(file_name=file_name)

    def save_features(self, features: pd.DataFrame, emb_func_name: str, graph_name: str,
                      removed_nodes: List[int], num_bins: int) -> None:
        file_name = self._get_feature_path_name(graph_name=graph_name, emb_func_name=emb_func_name,
                                                removed_nodes=removed_nodes, num_bins=num_bins)
        self.__save_csv_compressed(file_name=file_name, csv=features)

    def load_features(self, emb_func_name: str, graph_name: str, removed_nodes: List[int],
                      num_bins: int) -> pd.DataFrame:
        file_name = self._get_feature_path_name(graph_name=graph_name, emb_func_name=emb_func_name,
                                                removed_nodes=removed_nodes, num_bins=num_bins)
        return self.__load_compressed_csv(file_name=file_name)

    def load_training_features(self, emb_func_name: str, graph_name: str, attacked_node: int,
                               training_nodes: List[int], num_bins: int) -> pd.DataFrame:
        features = pd.DataFrame()
        for training_node in training_nodes:
            features = features.append(
                self.load_features(removed_nodes=[attacked_node, training_node], emb_func_name=emb_func_name,
                                   graph_name=graph_name, num_bins=num_bins))
        return features

    def save_classification_model(self, classification_model: sk_base.ClassifierMixin,
                                  attacked_node: int, training_nodes: List[int],
                                  emb_func_name: str, graph_name: str,
                                  num_bins: int) -> None:
        file_name = self._get_model_path_name(graph_name=graph_name, emb_func_name=emb_func_name,
                                              attacked_node=attacked_node, training_nodes=training_nodes,
                                              num_bins=num_bins,
                                              classifier_name=str(classification_model))
        self.__assure_path_exists(os.path.dirname(file_name))
        joblib.dump(classification_model, filename=file_name + '.joblib')

    def has_classification_model(self, classifier_name: str, emb_func_name: str, graph_name: str,
                                 attacked_node: int, training_nodes: List[int], num_bins: int) -> bool:
        file_name = self._get_model_path_name(graph_name=graph_name, emb_func_name=emb_func_name,
                                              attacked_node=attacked_node, training_nodes=training_nodes,
                                              num_bins=num_bins,
                                              classifier_name=classifier_name)
        return os.path.exists(file_name + '.joblib')

    def load_classification_model(self, classifier_name: str, emb_func_name: str, graph_name: str,
                                  attacked_node: int, training_nodes: List[int],
                                  num_bins: int) -> sk_base.ClassifierMixin:
        file_name = self._get_model_path_name(graph_name=graph_name, emb_func_name=emb_func_name,
                                              attacked_node=attacked_node, training_nodes=training_nodes,
                                              num_bins=num_bins,
                                              classifier_name=classifier_name)
        return joblib.load(filename=file_name + '.joblib')

    def save_test_results_per_node(self, results_per_node: pd.DataFrame,
                                   dict_attack_train_nodes: Dict[int, List[int]],
                                   classifier_name: str,
                                   emb_func_name: str, graph_name: str,
                                   num_bins: int) -> None:
        file_name = self._get_test_results_per_attacked_node_path_name(graph_name=graph_name,
                                                                       emb_func_name=emb_func_name,
                                                                       num_bins=num_bins,
                                                                       classifier_name=str(classifier_name),
                                                                       dict_attack_train_nodes=dict_attack_train_nodes)
        self.__assure_path_exists(os.path.dirname(file_name))
        results_per_node.to_csv(file_name + ".csv")

    def has_test_results_per_node(self, dict_attack_train_nodes: Dict[int, List[int]],
                                  classifier_name: str,
                                  emb_func_name: str, graph_name: str,
                                  num_bins: int) -> bool:
        file_name = self._get_test_results_per_attacked_node_path_name(graph_name=graph_name,
                                                                       emb_func_name=emb_func_name,
                                                                       num_bins=num_bins,
                                                                       classifier_name=str(classifier_name),
                                                                       dict_attack_train_nodes=dict_attack_train_nodes)
        return os.path.exists(file_name + ".csv")

    def load_test_results_per_node(self, dict_attack_train_nodes: Dict[int, List[int]],
                                   classifier_name: str,
                                   emb_func_name: str, graph_name: str,
                                   num_bins: int) -> bool:
        file_name = self._get_test_results_per_attacked_node_path_name(graph_name=graph_name,
                                                                       emb_func_name=emb_func_name,
                                                                       num_bins=num_bins,
                                                                       classifier_name=str(classifier_name),
                                                                       dict_attack_train_nodes=dict_attack_train_nodes)
        test_res = pd.read_csv(file_name + ".csv", index_col=0)
        test_res.columns = test_res.columns.astype(int)
        return test_res

    def save_aggregated_test_results(self, results_per_node: pd.DataFrame,
                                     dict_attack_train_nodes: Dict[int, List[int]],
                                     classifier_name: str,
                                     emb_func_name: str, graph_name: str,
                                     num_bins: int) -> None:
        file_name = self._get_agg_test_results_path_name(graph_name=graph_name,
                                                         emb_func_name=emb_func_name,
                                                         num_bins=num_bins,
                                                         classifier_name=str(classifier_name),
                                                         dict_attack_train_nodes=dict_attack_train_nodes)
        self.__assure_path_exists(os.path.dirname(file_name))
        results_per_node.to_csv(file_name + ".csv")

    def has_aggregated_test_results(self, dict_attack_train_nodes: Dict[int, List[int]],
                                    classifier_name: str,
                                    emb_func_name: str, graph_name: str,
                                    num_bins: int) -> bool:
        file_name = self._get_agg_test_results_path_name(graph_name=graph_name,
                                                         emb_func_name=emb_func_name,
                                                         num_bins=num_bins,
                                                         classifier_name=str(classifier_name),
                                                         dict_attack_train_nodes=dict_attack_train_nodes)
        return os.path.exists(file_name + ".csv")

    def load_aggregated_test_results(self, dict_attack_train_nodes: Dict[int, List[int]],
                                     classifier_name: str,
                                     emb_func_name: str, graph_name: str,
                                     num_bins: int) -> bool:
        file_name = self._get_agg_test_results_path_name(graph_name=graph_name,
                                                         emb_func_name=emb_func_name,
                                                         num_bins=num_bins,
                                                         classifier_name=str(classifier_name),
                                                         dict_attack_train_nodes=dict_attack_train_nodes)
        test_res = pd.read_csv(file_name + ".csv", index_col=0)
        test_res.columns = test_res.columns.astype(int)
        return test_res
