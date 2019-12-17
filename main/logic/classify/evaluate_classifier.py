from typing import List

import pandas as pd

import sklearn.base as sk_base
import sklearn.metrics as metrics
import entities.graphs.graph as g
import entities.embeddings.embedding as e
import logic.classify.compute_labels as cl
import memory_access.memory_access as ma


def __auc(labels: List[int], probabilites_for_class_one: List[float]):
    fpr, tpr, thresholds = metrics.roc_curve(labels, probabilites_for_class_one, pos_label=1)
    return metrics.auc(fpr, tpr)


def __create_ranking(labels: List[int], predicted: List[int], probabilities: List[float]):
    df = pd.DataFrame(data=[labels, predicted, probabilities], index=["labels", "predicted", "probabilities"]).T
    df = df.sort_values(by="probabilities", axis=0, ascending=False)
    return df


def __precision_at_k(labels: List[int], predicted: List[int], probabilities: List[float], k: int):
    ranking = __create_ranking(labels, predicted, probabilities)
    ranking = ranking.head(n=k)
    return metrics.precision_score(y_true=ranking["labels"].tolist(), y_pred=ranking["predicted"].tolist(), pos_label=1)


def __reciprocal_rank(labels: List[int], predicted: List[int], probabilities: List[float]):
    ranking = __create_ranking(labels, predicted, probabilities)
    first_pos_index = ranking["labels"].tolist().index(1)
    return 1 / (first_pos_index + 1)


def __harmonic_mean(precision: int, recall: int):
    return 2 * (precision * recall) / (precision + recall)


def evaluate(test_labels: List[bool], predicted_labels: List[bool], predicted_prob: List[float],
             attacked_node: int, training_nodes: List[int]) -> pd.DataFrame:
    assert (len(test_labels) == len(predicted_labels))
    assert (len(predicted_labels) == len(predicted_prob))

    tn, fp, fn, tp = metrics.confusion_matrix(test_labels, predicted_labels).ravel()

    evaluation = pd.DataFrame(columns=[attacked_node])

    evaluation.loc["training nodes"] = str(training_nodes)

    evaluation.loc["true negative"] = tn
    evaluation.loc["false positive"] = fp
    evaluation.loc["false negative"] = fn
    evaluation.loc["true positive"] = tp

    evaluation.loc["accuracy"] = metrics.accuracy_score(test_labels, predicted_labels)
    evaluation.loc["precision"] = metrics.precision_score(test_labels, predicted_labels, pos_label=1)
    evaluation.loc["recall"] = metrics.recall_score(test_labels, predicted_labels, pos_label=1)
    evaluation.loc["true negative rate"] = tn / (tn + fp)

    evaluation.loc["precision at 5"] = __precision_at_k(labels=test_labels, predicted=predicted_labels,
                                                        probabilities=predicted_prob, k=5)
    evaluation.loc["precision at 10"] = __precision_at_k(labels=test_labels, predicted=predicted_labels,
                                                         probabilities=predicted_prob, k=10)
    evaluation.loc["precision at 20"] = __precision_at_k(labels=test_labels, predicted=predicted_labels,
                                                         probabilities=predicted_prob, k=20)

    evaluation.loc["auc"] = __auc(labels=test_labels, probabilites_for_class_one=predicted_prob)
    evaluation.loc["binary f1"] = metrics.f1_score(y_true=test_labels, y_pred=predicted_labels)

    evaluation.loc["reciprocal rank"] = __reciprocal_rank(labels=test_labels, predicted=predicted_labels,
                                                          probabilities=predicted_prob)

    return evaluation


def evaluate_classifier(trained_classifier: sk_base.ClassifierMixin, attacked_node: int, training_nodes: List[int],
                        graph: g.Graph,
                        embedding: e.Embedding,
                        num_bins: int,
                        mem_acc: ma.MemoryAccess) -> pd.DataFrame:
    """
    Evaluates a trained classification model and computes metrics
    :param trained_classifier: model which is evaluated
    :param attacked_node: node whos neighbors the model should predict
    :param training_nodes: nodes used to generate training Scenarios to train the model
    :param graph: graph the experiment is based on
    :param embedding: embedding the experiment is based on
    :param num_bins: number of bins used for the feature computation
    :param mem_acc: obj for memory access
    :return: evaluation of the model
    """
    test_features = mem_acc.load_features(emb_func_name=str(embedding), graph_name=str(graph),
                                          removed_nodes=[attacked_node],
                                          num_bins=num_bins)
    test_labels = cl.compute_labels(node_list=list(test_features.index), graph=graph, removed_node=attacked_node)

    predicted_labels = trained_classifier.predict(test_features.values)
    predicted_prob = trained_classifier.predict_proba(test_features.values)[:, 1]

    return evaluate(test_labels=test_labels, predicted_labels=predicted_labels, predicted_prob=predicted_prob,
                    attacked_node=attacked_node, training_nodes=training_nodes)


def aggregate_evaluations(evaluations: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates classification model evaluations over multiple evaluations and computes metrics
    :param evaluations: each column of the DataFrame contains the evaluation of one model
    :return: Evaluation over all input evaluations
    """
    if not (len(evaluations.columns) == len(set(evaluations.columns))):
        raise ValueError(f"Not all evaluations are of different nodes! "
                         f"Evaluations {evaluations} (duplicate column header)")

    agg_eval = pd.DataFrame(columns=[str(evaluations.columns)])

    # add information about the attacked nodes and used training nodes
    tr_info: str = ""
    for col in evaluations.columns:
        tr_info += f"{col}{evaluations[col]['training nodes']},"
    agg_eval.loc["training nodes"] = tr_info

    agg_eval.loc["true positive"] = evaluations.loc["true positive"].sum()
    agg_eval.loc["false positive"] = evaluations.loc["false positive"].sum()
    agg_eval.loc["false negative"] = evaluations.loc["false negative"].sum()
    agg_eval.loc["true negative"] = evaluations.loc["true negative"].sum()

    agg_eval.loc["macro accuracy"] = evaluations.loc["accuracy"].mean()
    agg_eval.loc["macro precision"] = evaluations.loc["precision"].mean()
    agg_eval.loc["macro recall"] = evaluations.loc["recall"].mean()
    agg_eval.loc["macro true negative rate"] = evaluations.loc["true negative rate"].mean()

    agg_eval.loc["macro precision at 5"] = evaluations.loc["precision at 5"].mean()
    agg_eval.loc["macro precision at 10"] = evaluations.loc["precision at 10"].mean()
    agg_eval.loc["macro precision at 20"] = evaluations.loc["precision at 20"].mean()

    agg_eval.loc["micro accuracy"] = (agg_eval.loc["true positive"] + agg_eval.loc[
        "true negative"]) / (agg_eval.loc["true positive"] + agg_eval.loc["true negative"] +
                             agg_eval.loc["false positive"] + agg_eval.loc["false negative"])
    agg_eval.loc["micro precision"] = agg_eval.loc["true positive"] / (
            agg_eval.loc["true positive"] + agg_eval.loc["false positive"])
    agg_eval.loc["micro recall"] = agg_eval.loc["true positive"] / (
            agg_eval.loc["true positive"] + agg_eval.loc["false negative"])
    agg_eval.loc["micro true negative rate"] = agg_eval.loc["true negative"] / (
            agg_eval.loc["true negative"] + agg_eval.loc["false positive"])

    agg_eval.loc["avg auc"] = evaluations.loc["auc"].mean()
    agg_eval.loc["macro f1"] = __harmonic_mean(agg_eval.loc["macro precision"], agg_eval.loc["macro recall"])
    agg_eval.loc["micro f1"] = __harmonic_mean(agg_eval.loc["micro precision"], agg_eval.loc["micro recall"])
    agg_eval.loc["mean reciprocal rank"] = evaluations.loc["reciprocal rank"].mean()

    return agg_eval
