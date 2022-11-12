import unittest

from network.synapse import Synapse


class TestSynapse(unittest.TestCase):
    def test_synapse_with_weight(self):
        synapse = Synapse(0, 1, weight=5)
        self.assertEqual(5, synapse.weight)

    def test_synapse_should_have_from_and_to(self):
        synapse = Synapse(0, 1)
        self.assertEqual(0, synapse.connect_from)
        self.assertEqual(1, synapse.connect_to)
