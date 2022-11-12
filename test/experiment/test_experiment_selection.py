import unittest

from experiment.brian.classification import Classification, ClassificationTask
from experiment.brian.xor import XOR
from experiment.experiment_selection import ExperimentSelection
from utility.configuration import Configuration


class TestExperimentSelection(unittest.TestCase):
    def test_undefined_experiment(self):
        c = Configuration({"experiment": "abc"})
        selection = ExperimentSelection(c)

        self.assertRaises(RuntimeError, selection.get_experiment)

    def test_xor_brian(self):
        c = Configuration({"experiment": "xor"})
        selection = ExperimentSelection(c)

        experiment = selection.get_experiment()

        self.assertTrue(isinstance(experiment, XOR))

    def test_classification_parameters(self):
        c = Configuration(
            {
                "experiment": "classification",
                "experiment_options": {
                    "task": ClassificationTask.BREAST,
                    "rounds": 5,
                },
            }
        )
        selection = ExperimentSelection(c)

        experiment: Classification = selection.get_experiment()

        self.assertTrue(isinstance(experiment, Classification))

        self.assertEqual(experiment.rounds, 5)
        self.assertEqual(experiment.task, ClassificationTask.BREAST)
        self.assertEqual(len(experiment.X_train), 455)

    def test_classification_task_string(self):
        c = Configuration(
            {
                "experiment": "classification",
                "experiment_options": {"task": "breast"},
            }
        )
        selection = ExperimentSelection(c)

        experiment = selection.get_experiment()
        self.assertEqual(experiment.task, ClassificationTask.BREAST)
