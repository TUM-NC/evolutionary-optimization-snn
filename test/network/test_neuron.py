import unittest

from network.neuron import Neuron


class TestNeuron(unittest.TestCase):
    def test_neuron_with_threshold(self):
        neuron = Neuron(uid=1, threshold=10)
        self.assertEqual(10, neuron.threshold)

    def test_neuron_without_threshold(self):
        neuron = Neuron(uid=1)
        self.assertTrue("threshold" not in neuron.parameters)
        # expected behavior, when attribute not defined at first
        self.assertRaises(AttributeError, neuron.__getattr__, "threshold")

    def test_neuron_should_have_uid(self):
        neuron = Neuron(uid=1)
        self.assertEqual(1, neuron.uid)
