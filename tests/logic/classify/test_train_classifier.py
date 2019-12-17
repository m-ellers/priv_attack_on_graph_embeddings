from unittest import TestCase

import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

import main.logic.classify.train_classifier as tc


class TestTrainClassifier(TestCase):

    def setUp(self) -> None:
        self.classifiers = [
            KNeighborsClassifier(n_neighbors=2),
            SVC(kernel="linear", probability=True),
            DecisionTreeClassifier(),
            RandomForestClassifier(),
            AdaBoostClassifier(),
            GaussianNB(),
            MLPClassifier(max_iter=100000, hidden_layer_sizes=(100, 100,))]

    def test_fit_classifier(self):
        for classifier in self.classifiers:
            with self.subTest(str(classifier)):
                # given
                training_features = pd.DataFrame([[0.4, 0.1, 0.0, 0.5, 0.7],
                                                  [0.1, 0.3, 0.6, 0.1, 0.4],
                                                  [0.0, 0.6, 0.3, 0.1, 1],
                                                  [0.7, 0.0, 0.0, 0.3, 0.1]], index=[0, 1, 2, 3])
                training_labels = pd.Series([True, False, False, True], index=[0, 1, 2, 3])

                # when
                c_model = tc.fit_classifier(training_features=training_features, training_labels=training_labels,
                                            classifier=classifier)

                # then
                self.assertTrue(c_model.predict(np.array([training_features.loc[0]]))[0],
                                msg=f"True training data {training_features.loc[0]} is misclassified")
                self.assertTrue(c_model.predict(np.array([training_features.loc[3]]))[0],
                                msg=f"True training data {training_features.loc[3]} is misclassified")
                self.assertFalse(c_model.predict(np.array([training_features.loc[1]]))[0],
                                 msg=f"False training data {training_features.loc[1]} is misclassified")
                self.assertFalse(c_model.predict(np.array([training_features.loc[2]]))[0],
                                 msg=f"False training data {training_features.loc[2]} is misclassified")

                self.assertTrue(c_model.predict(np.array([[0.5, 0.1, 0.05, 0.45, 0.4]]))[0],
                                msg=f"Misclassified test data {[0.5, 0.1, 0.05, 0.45, 0.4]}")
                self.assertFalse(c_model.predict(np.array([[0.04, 0.76, 0.2, 0.0, 0.5]]))[0],
                                 msg=f"Misclassified test data {[0.04, 0.76, 0.2, 0.0, 0.5]}")
