from typing import Dict, Optional, Type

from experiment.brian.cart_pole_balancing import CartPoleBalancing
from experiment.brian.classification import Classification
from experiment.brian.xor import XOR
from experiment.dummy import Dummy
from experiment.experiment import Experiment
from utility.configurable import Configurable
from utility.configuration import Configuration

experiment_mapping: Dict[str, Type[Experiment]] = {
    "xor": XOR,
    "cart_pole": CartPoleBalancing,
    "classification": Classification,
    "dummy": Dummy,
}


def valid_experiment(s):
    """
    whether the given key is in the experiment mapping
    :param s:
    :return:
    """
    return s in experiment_mapping


class ExperimentSelection(Configurable):
    # configurable attributes
    experiment: str = None
    experiment_options: dict = {}

    def __init__(
        self,
        configuration: Optional[Configuration] = None,
    ):
        super().__init__(configuration=configuration)

    def get_experiment(self) -> Experiment:
        """
        Get the experiment for  running

        :return:
        """
        if not valid_experiment(self.experiment):
            raise RuntimeError(
                f"The specified experiment '{self.experiment}' is not defined,"
                f" use any of: {list(experiment_mapping.keys())}"
            )

        experiment_type = experiment_mapping[self.experiment]

        # if experiment options are not given, interpret it as empty options
        options = self.experiment_options
        if options is None:
            options = {}

        return experiment_type(**options)

    def set_configurable(self):
        self.add_configurable_attribute(
            "experiment", "Select experiment to run", validate=valid_experiment
        )

        self.add_configurable_attribute(
            "experiment_options", "Select experiment to run"
        )
