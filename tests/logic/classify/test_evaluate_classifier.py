from unittest import TestCase

import numpy as np
import pandas as pd

import main.logic.classify.evaluate_classifier as ec


class TestEvaluateClassifier(TestCase):
    def test_evaluate(self):
        # given
        labels = [True, False, True, True, False, True]
        predicted_labels = [True, False, False, True, True, False]
        predicted_proba = [0.7, 0.3, 0.4, 0.9, 0.55, 0.2]
        attacked_node = 42
        # when
        evaluation = ec.evaluate(test_labels=labels, predicted_labels=predicted_labels, predicted_prob=predicted_proba,
                                 attacked_node=attacked_node, training_nodes=[0, 1, 2])

        # then
        with self.subTest("true negative"):
            np.testing.assert_almost_equal(evaluation.loc["true negative"].values, 1)
        with self.subTest("false positive"):
            np.testing.assert_almost_equal(evaluation.loc["false positive"].values, 1)
        with self.subTest("false negative"):
            np.testing.assert_almost_equal(evaluation.loc["false negative"].values, 2)
        with self.subTest("true positive"):
            np.testing.assert_almost_equal(evaluation.loc["true positive"].values, 2)
        with self.subTest("accuracy"):
            np.testing.assert_almost_equal(evaluation.loc["accuracy"].values, 3 / 6)
        with self.subTest("precision"):
            np.testing.assert_almost_equal(evaluation.loc["precision"].values, 2 / 3)
        with self.subTest("recall"):
            self.assertEqual(evaluation.loc["recall"].values, 2 / 4)
        with self.subTest("true negative rate"):
            self.assertEqual(evaluation.loc["true negative rate"].values, 1 / 2)
        with self.subTest("precision at 5"):
            np.testing.assert_almost_equal(evaluation.loc["precision at 5"].values, 2 / 3)
        with self.subTest("precision at 10"):
            np.testing.assert_almost_equal(evaluation.loc["precision at 10"].values, 2 / 3)
        with self.subTest("precision at 20"):
            np.testing.assert_almost_equal(evaluation.loc["precision at 20"].values, 2 / 3)
        with self.subTest("auc"):
            np.testing.assert_almost_equal(evaluation.loc["auc"].values, 5 / 8)
        with self.subTest("binary f1"):
            np.testing.assert_almost_equal(evaluation.loc["binary f1"].values, 4 / 7)
        with self.subTest("reciprocal rank"):
            self.assertEqual(evaluation.loc["reciprocal rank"].values, 1)

    def test_concat_evaluations(self):
        evaluation_one = pd.DataFrame(columns=[1])
        evaluation_one.loc["training nodes"] = str([0, 3, 2])
        evaluation_one.loc["true negative"] = 100
        evaluation_one.loc["false positive"] = 10
        evaluation_one.loc["false negative"] = 3
        evaluation_one.loc["true positive"] = 7
        evaluation_one.loc["accuracy"] = 0.5
        evaluation_one.loc["precision"] = 0.7
        evaluation_one.loc["recall"] = 0.3
        evaluation_one.loc["true negative rate"] = 0.9
        evaluation_one.loc["precision at 5"] = 0.8
        evaluation_one.loc["precision at 10"] = 0.6
        evaluation_one.loc["precision at 20"] = 0.7
        evaluation_one.loc["auc"] = 0.7
        evaluation_one.loc["binary f1"] = 0.7
        evaluation_one.loc["reciprocal rank"] = 1

        evaluation_two = pd.DataFrame(columns=[0])
        evaluation_two.loc["training nodes"] = str([1, 3, 2])
        evaluation_two.loc["true negative"] = 50
        evaluation_two.loc["false positive"] = 40
        evaluation_two.loc["false negative"] = 7
        evaluation_two.loc["true positive"] = 9
        evaluation_two.loc["accuracy"] = 0.5
        evaluation_two.loc["precision"] = 0.8
        evaluation_two.loc["recall"] = 0.5
        evaluation_two.loc["true negative rate"] = 0.95
        evaluation_two.loc["precision at 5"] = 0.8
        evaluation_two.loc["precision at 10"] = 0.4
        evaluation_two.loc["precision at 20"] = 0.9
        evaluation_two.loc["auc"] = 0.2
        evaluation_two.loc["binary f1"] = 0.2
        evaluation_two.loc["reciprocal rank"] = 0.5

        # when
        evaluations_df = pd.concat([evaluation_one, evaluation_two], axis=1)
        combination = ec.aggregate_evaluations(evaluations_df)

        # then
        with self.subTest("true negative"):
            np.testing.assert_almost_equal(combination.loc["true negative"].values, 150)
        with self.subTest("false positive"):
            np.testing.assert_almost_equal(combination.loc["false positive"].values, 50)
        with self.subTest("false negative"):
            np.testing.assert_almost_equal(combination.loc["false negative"].values, 10)
        with self.subTest("true positive"):
            np.testing.assert_almost_equal(combination.loc["true positive"].values, 16)

        with self.subTest("macro accuracy"):
            np.testing.assert_almost_equal(combination.loc["macro accuracy"].values, 0.5)
        with self.subTest("macro precision"):
            np.testing.assert_almost_equal(combination.loc["macro precision"].values, 0.75)
        with self.subTest("macro recall"):
            self.assertEqual(combination.loc["macro recall"].values, 0.4)
        with self.subTest("macro true negative rate"):
            self.assertEqual(combination.loc["macro true negative rate"].values, 0.925)
        with self.subTest("macro precision at 5"):
            np.testing.assert_almost_equal(combination.loc["macro precision at 5"].values, 0.8)
        with self.subTest("macro precision at 10"):
            np.testing.assert_almost_equal(combination.loc["macro precision at 10"].values, 0.5)
        with self.subTest("macro precision at 20"):
            np.testing.assert_almost_equal(combination.loc["macro precision at 20"].values, 0.8)

        with self.subTest("mico accuracy"):
            np.testing.assert_almost_equal(combination.loc["micro accuracy"].values, 166 / 226)
        with self.subTest("mico precision"):
            np.testing.assert_almost_equal(combination.loc["micro precision"].values, 16 / 66)
        with self.subTest("mico recall"):
            self.assertEqual(combination.loc["micro recall"].values, 16 / 26)
        with self.subTest("mico true negative rate"):
            self.assertEqual(combination.loc["micro true negative rate"].values, 150 / 200)
        with self.subTest("avg auc"):
            np.testing.assert_almost_equal(combination.loc["avg auc"].values, 0.45)
        with self.subTest("macro f1"):
            np.testing.assert_almost_equal(combination.loc["macro f1"].values, 0.6 / 1.15)
        with self.subTest("micro f1"):
            np.testing.assert_almost_equal(combination.loc["micro f1"].values, 8 / 23)
        with self.subTest("mean reciprocal rank"):
            np.testing.assert_almost_equal(combination.loc["mean reciprocal rank"].values, 0.75)
