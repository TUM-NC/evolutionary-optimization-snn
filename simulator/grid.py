"""
A class to for multiprocessing of multiple experiments
"""
import csv
from multiprocessing import Pool, Value
from pathlib import Path
from typing import Dict, List, Optional, Union

from experiment.experiment_selection import ExperimentSelection
from network.evolution.framework import Framework
from utility.configuration import Configuration
from utility.flatten_dict import flatten_dict
from utility.grid_configuration import GridConfiguration

# to share a variable between processes
shared_count = None


def init(args):
    """
    Function to initialize shared variable between processes
    :param args:
    :return:
    """
    global shared_count
    shared_count = args


def run_option(option):
    """
    Run an option in multi-tasking

    :param option:
    :return:
    """

    def run_experiment():
        """
        Run an experiment, return stats and time taken
        :return:
        """
        experiment_selection = ExperimentSelection(configuration=c)
        experiment = experiment_selection.get_experiment()

        f = Framework(experiment=experiment, configuration=c)
        s = f.evolution()

        return s

    (index, c, save) = option
    save: Optional[str]

    # get a custom counter for the progress output
    # -> Python pool uses custom order
    global shared_count
    with shared_count.get_lock():
        shared_count.value += 1
        pool_index = shared_count.value

    print(f"============= Test option {pool_index} =================")

    config: Configuration
    stats = run_experiment()

    if save is not None:
        p = Path(save).joinpath(f"stats_{index}.json")
        stats.to_file(p, None)

    _, best_fitness = stats.get_best_network_alltime()
    epochs = stats.get_amount_epochs()
    took = stats.get_total_time_took()

    print(
        f"============= Test option {pool_index} finished "
        f" - Fitness: {best_fitness}"
        f" - Epochs: {epochs}"
        f" - Took: {round(took, 2)}s ================="
    )

    return {
        "index": index,
        "best_fitness": best_fitness,
        "epochs": epochs,
        "took": took,
    }


class GridSimulator:
    _computations: List[Dict[str, Union[int, None, float]]] = []

    def __init__(self, grid_configuration: GridConfiguration):
        self.grid_configuration = grid_configuration

        if grid_configuration.save is not None:
            path = Path(grid_configuration.save)
            path.mkdir(parents=True, exist_ok=True)

    def iterate(self):
        """
        Iterate through grid_configurations via python multiprocessing

        :return:
        """
        iterations = self.grid_configuration.get_amount_alternatives()
        global shared_count
        shared_count = Value("i", 0)

        iterable = [
            (
                i,
                self.grid_configuration.get_option(i),
                self.grid_configuration.save,
            )
            for i in range(iterations)
        ]

        p = Pool(
            processes=self.grid_configuration.pool_size,
            initargs=(shared_count,),
            initializer=init,
        )
        self._computations = p.map(run_option, iterable)

        if self.grid_configuration.save is not None:
            save = Path(self.grid_configuration.save)
            csv_path = save.joinpath("grid.csv")
            headers, values = self.get_table_values()

            with open(csv_path, "w") as f:
                writer = csv.writer(f, delimiter=",", quotechar='"')
                writer.writerow(headers)
                writer.writerows(values)

            base = save.joinpath("config.yaml")
            grid = save.joinpath("grid.yaml")

            self.grid_configuration.to_yaml(base, grid)

    def get_computations(self):
        """
        Get results from computation

        :return:
        """
        return self._computations

    def get_table_values(self, empty_value="-"):
        """
        return headers and values for grid computation

        :param empty_value:
        :return: tuple with header, values
        """
        option_headers = self.grid_configuration.get_option_headers()
        # all computations should have same format -> use first element
        exclude = []
        additional_headers = list(
            filter(lambda a: a not in exclude, self._computations[0].keys())
        )
        all_headers = option_headers + additional_headers

        values = []

        for c in self._computations:
            i = c["index"]

            option = flatten_dict(
                self.grid_configuration.get_option(i).get_config_dict()
            )

            value = []
            for header in option_headers:
                if header in option:
                    value.append(option[header])
                else:
                    value.append(empty_value)

            for header in additional_headers:
                value.append(c[header])

            values.append(value)

        return all_headers, values
