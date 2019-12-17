from typing import List, Dict

import main.entities.embeddings.embedding as emb

import main.entities.graphs.graph as g
import sklearn.base as sk_base
import sklearn.naive_bayes as sk_nb


class Parameters():
    def __init__(self, graph: g.Graph,
                 embedding: emb.Embedding,
                 dict_attacked_nodes_training_nodes: Dict[int, List[int]],
                 num_of_bins: int = 10,
                 classifier: sk_base.ClassifierMixin = sk_nb.GaussianNB()
                 ):
        """
        Entity Class which saves all parametrisation information for the experiments
        :param graph: graph the experiments are performed on
        :param embedding: embedding used for exp.
        :param dict_attacked_nodes_training_nodes: dictionary that contains the attacked(removed) nodes for the exp.
            as keys and their respective training nodes which are additionally removed from the network to compute
            training data.
        :param num_of_bins: number of bins for feature computation
        :param classifier: classifier used on the features to predict neighbors of the attacked node
        """
        self.graph = graph
        self.dict_attacked_nodes_training_nodes = dict_attacked_nodes_training_nodes
        self.embedding = embedding
        self.num_of_bins = num_of_bins
        self.classifier = classifier
