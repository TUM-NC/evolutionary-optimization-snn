import unittest

from network.decoder.brian.classification import ClassificationBrianDecoder


class TestClassificationBrianDecoder(unittest.TestCase):
    def test_two_classes(self):
        decoder = ClassificationBrianDecoder(classes=2)

        self.assertEqual(0, decoder.get_best_class([1, 0]))
        self.assertEqual(1, decoder.get_best_class([0, 1]))
        self.assertEqual(-1, decoder.get_best_class([0, 0]))
        self.assertEqual(-1, decoder.get_best_class([1, 1]))

    def test_four_classes(self):
        decoder = ClassificationBrianDecoder(classes=4)

        self.assertEqual(2, decoder.get_best_class([0, 1, 2, 0]))
        self.assertEqual(3, decoder.get_best_class([0, 1, 2, 3]))
        self.assertEqual(-1, decoder.get_best_class([0, 1, 2, 2]))
        self.assertEqual(-1, decoder.get_best_class([0, 0, 0, 0]))
        self.assertEqual(3, decoder.get_best_class([1, 1, 1, 2]))
