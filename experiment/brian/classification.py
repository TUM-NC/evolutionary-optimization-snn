from enum import Enum
from typing import List

from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from experiment.brian.brian_experiment import BrianExperiment
from network.decoder.brian.classification import ClassificationBrianDecoder
from network.encoder.brian.float import FloatBrianEncoder
from network.network import Network


class ClassificationTask(Enum):
    IRIS = "iris"
    BREAST = "breast"
    WINE = "wine"


def normalize(data):
    """
    Normalize and standize a given data set

    :param data:
    :return:
    """
    std_scaler = StandardScaler()
    data = std_scaler.fit_transform(data)

    min_max_scaler = MinMaxScaler()
    data = min_max_scaler.fit_transform(data)
    return data


def get_data_set(task: ClassificationTask):
    """
    Return the data set for the given task

    :param task:
    :return:
    """
    if task == ClassificationTask.IRIS:
        return datasets.load_iris()
    if task == ClassificationTask.BREAST:
        return datasets.load_breast_cancer()
    if task == ClassificationTask.WINE:
        return datasets.load_wine()

    raise RuntimeError("This classification task is not implemented")


class Classification(BrianExperiment):
    """
    Actual experiment implementation using brian for simulation
    """

    X_train: List[tuple]
    y_train: List[int]
    X_test: List[tuple]
    y_test: List[int]

    encoder: FloatBrianEncoder
    decoder: ClassificationBrianDecoder

    rounds: int  # train same sample this many times
    task: ClassificationTask

    penalize_network_size: bool

    def __init__(
        self,
        task=ClassificationTask.IRIS,
        train_size=0.8,
        rounds=1,
        split_seed=1,
        poisson=True,
        penalize_network_size=False,
    ):
        """
        :param task: classification task
        :param train_size: can be int or float from 0 to 1
        :param rounds: train multiple times on each training sample
        """
        # if string is given, should convert to ClassificationTask
        if isinstance(task, str):
            task = ClassificationTask(task)

        dataset = get_data_set(task)
        X = dataset.data
        y = dataset.target

        input_neurons = X.shape[1]
        classes = dataset.target_names.size

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, random_state=split_seed
        )

        # normalize train and test separately
        self.X_train = [tuple(values) for values in normalize(X_train)]
        self.y_train = list(y_train)
        self.X_test = [tuple(values) for values in normalize(X_test)]
        self.y_test = list(y_test)

        self.encoder = FloatBrianEncoder(
            number_of_neurons=input_neurons, poisson=poisson
        )
        self.decoder = ClassificationBrianDecoder(classes=classes)

        self.rounds = rounds
        self.task = task
        self.penalize_network_size = penalize_network_size

    def simulate(self, networks: List[Network]):
        input_patterns = self.X_train * self.rounds
        calculated = self._simulate_on_multiple_inputs(
            networks, input_patterns
        )

        for i, network in enumerate(networks):
            self.set_output_by_network(network, calculated[i])

    def single_fitness(self, network: Network):
        """
        Calculate the fitness values for a single network

        :param network:
        :return:
        """
        calculated = self.get_output_by_network(network)
        classifications = [v[0] for v in calculated]
        expected = self.y_train * self.rounds
        correct_classifications = self.get_correct_classifications(
            classifications, expected
        ) / len(expected)
        if self.penalize_network_size:
            correct_classifications = correct_classifications - (
                0.00001 * (len(network.hidden_neurons) + len(network.synapses))
            )
        return correct_classifications

    @staticmethod
    def get_correct_classifications(values, target):
        """
        Get the number of correct classifications

        :param values: network outputs
        :param target: expected outputs
        :return:
        """
        return sum(
            1
            for index, classification in enumerate(values)
            if target[index] == classification
        )

    @staticmethod
    def get_confusion_matrix(values, target):
        """
        Get a confusion matrix based on the given values
        :param values: y_pred
        :param target: y_true
        :return:
        """
        return confusion_matrix(target, values)

    def test_network(self, network: Network, runs=1, on_test_set=True):
        """
        Run the given network on the test set
        return amount of correct classifications

        :param network:
        :param runs: how often should the test set be tested
        :return:
        """
        if on_test_set:
            test_set = self.X_test * runs
            expected = list(self.y_test) * runs
        else:
            test_set = self.X_train * runs
            expected = list(self.y_train) * runs

        outputs = self._simulate_on_multiple_inputs([network], test_set)
        output = outputs[0]  # we only simulate a single network
        classifications = [v[0] for v in output]

        correct_classifications = self.get_correct_classifications(
            classifications, expected
        )
        confusion = self.get_confusion_matrix(classifications, expected)
        classifications_per_run = correct_classifications / runs
        percent = correct_classifications / len(test_set)
        return {
            "total": correct_classifications,
            "per_run": classifications_per_run,
            "percent_correct": percent,
            "confusion_matrix": confusion,
        }
