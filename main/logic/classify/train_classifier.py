from typing import List

import pandas as pd

import sklearn.base as sk_base

import entities.graphs.graph as g
import entities.embeddings.embedding as e
import logic.classify.compute_labels as cl
import memory_access.memory_access as ma


def fit_classifier(training_features: pd.DataFrame, training_labels: pd.Series, classifier: sk_base.ClassifierMixin):
    assert (all(training_features.index == training_labels.index))
    classifier.fit(training_features.values, training_labels.values)
    return classifier


def load_train_features_and_labels(attacked_node: int, training_nodes: List[int], embedding: e.Embedding,
                                   graph_p: g.Graph, num_bins: int, mem_acc: ma.MemoryAccess,
                                   ) -> (pd.DataFrame, pd.Series):
    train_features = pd.DataFrame()
    train_labels = pd.Series()

    for train_node in training_nodes:
        h_train_features = mem_acc.load_features(emb_func_name=str(embedding), graph_name=str(graph_p),
                                                 removed_nodes=[attacked_node, train_node], num_bins=num_bins)
        h_train_labels = cl.compute_labels(node_list=list(h_train_features.index), graph=graph_p,
                                           removed_node=train_node)

        assert (all(h_train_features.index == h_train_labels.index))
        train_features = train_features.append(h_train_features)
        train_labels = train_labels.append(h_train_labels)

    assert (len(train_features.columns) == num_bins + 1)  # bin columns + deg column
    return train_features, train_labels


def train_classifier(attacked_node: int, training_nodes: List[int], graph_p: g.Graph, embedding: e.Embedding,
                     num_bins: int, classifier: sk_base.ClassifierMixin,
                     mem_acc: ma.MemoryAccess) -> sk_base.ClassifierMixin:
    """
    Loads training features and computes a classification model from it to predict if a node was a neighbor of the
        removed node 'attacked_node' in the original graph
    :param attacked_node: node which is removed from graph_p
    :param training_nodes: training features are computed from removing these nodes from graph_p
    :param graph_p: graph without node 'attacked_node'
    :param embedding: embedding fundction which is used to compute the embeddings
    :param num_bins: number of bins used for the feature computation
    :param classifier: Classifier trained in this method. Must have methods fit(), predict() and predict_proba()
    :param mem_acc: obj for memory access
    :return: trained classifier
    """
    assert (attacked_node not in graph_p.nodes())
    assert (all([second_rem_node in graph_p.nodes() for second_rem_node in training_nodes]))

    if mem_acc.has_classification_model(classifier_name=str(classifier), emb_func_name=str(embedding),
                                        graph_name=str(graph_p), attacked_node=attacked_node,
                                        training_nodes=training_nodes, num_bins=num_bins):
        return mem_acc.load_classification_model(classifier_name=str(classifier), emb_func_name=str(embedding),
                                                 graph_name=str(graph_p), attacked_node=attacked_node,
                                                 training_nodes=training_nodes, num_bins=num_bins)

    train_features, train_labels = load_train_features_and_labels(attacked_node=attacked_node,
                                                                  training_nodes=training_nodes,
                                                                  embedding=embedding, graph_p=graph_p,
                                                                  num_bins=num_bins, mem_acc=mem_acc)
    classifier = fit_classifier(training_features=train_features, training_labels=train_labels, classifier=classifier)

    mem_acc.save_classification_model(classification_model=classifier, emb_func_name=str(embedding),
                                      graph_name=str(graph_p), attacked_node=attacked_node,
                                      training_nodes=training_nodes, num_bins=num_bins)

    return classifier
