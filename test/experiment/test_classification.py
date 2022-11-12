import unittest

import numpy as np

from experiment.brian.classification import Classification


class TestClassification(unittest.TestCase):
    def test_correct_amount(self):
        values = [0, 1, 2, 3, 1, 2, 2]
        target = [0, 0, 3, 2, 1, 2, 1]

        calculate = Classification.get_correct_classifications(values, target)
        self.assertEqual(3, calculate)

    def test_confusion_matrix(self):
        values = [0, 1, 2, 3, 1, 2, 2]
        target = [0, 0, 3, 2, 1, 2, 1]

        matrix = Classification.get_confusion_matrix(values, target)

        assumed_matrix = [
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 0],
        ]
        self.assertEqual(assumed_matrix, matrix.tolist())

    def test_confusion_diagonal_correct_match(self):
        """
        The diagonal of the matrix, should be same as:
        get_correct_classifications
        :return:
        """
        values = np.random.randint(5, size=20)
        target = np.random.randint(5, size=20)

        matrix = Classification.get_confusion_matrix(values, target)
        diagonal = sum(np.diagonal(matrix))

        calculate = Classification.get_correct_classifications(values, target)
        self.assertEqual(calculate, diagonal)
