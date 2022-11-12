import unittest

from network.decoder.brian.binary import BinaryBrianDecoder


class TestBinaryBrianDecoder(unittest.TestCase):
    def test_absolute_distance(self):
        decoder = BinaryBrianDecoder(boundary=75, ideal_distance=25)

        self.assertEqual(0, decoder.get_absolute_distance(50, False))
        self.assertEqual(0, decoder.get_absolute_distance(100, True))
        self.assertEqual(25, decoder.get_absolute_distance(75, True))
        self.assertEqual(25, decoder.get_absolute_distance(75, False))
        self.assertEqual(75, decoder.get_absolute_distance(125, False))
        self.assertEqual(25, decoder.get_absolute_distance(125, True))
