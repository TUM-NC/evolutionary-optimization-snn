import argparse
import os
import warnings

import tikzplotlib

from experiment.brian.classification import Classification
from experiment.experiment_selection import ExperimentSelection
from network.evolution.framework import Framework
from network.evolution.stats import Stats
from utility.configuration import Configuration

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a neuro-evolution experiment"
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Provide a yaml file for configuration",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--no-plot",
        help="Don't show any plots",
        dest="plot",
        action="store_false",
    )
    parser.add_argument(
        "--save-plot",
        help="Save plots to files",
        dest="save_plot",
        default=None,
    )
    parser.add_argument(
        "-b",
        "--no-best",
        help="Show best network",
        dest="best",
        action="store_false",
    )
    parser.add_argument(
        "-s", "--stats", help="Save stats", type=str, default=None
    )
    parser.add_argument(
        "-l",
        "--load-stats",
        help="Load stats, allows to resume a previous simulation",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--no-evolution",
        help="Don't perform further evolution",
        dest="evolution",
        action="store_false",
    )
    parser.add_argument(
        "-t",
        "--test",
        help="Test network (for classification)",
        dest="test",
        action="store_true",
    )
    parser.add_argument(
        "--test-runs", help="Rounds for test", type=int, default=None
    )
    args = parser.parse_args()

    if args.config:
        c = Configuration.from_yaml(args.config)
    else:
        c = Configuration()

    experiment_selection = ExperimentSelection(configuration=c)
    experiment = experiment_selection.get_experiment()

    f = Framework(experiment=experiment, configuration=c)
    c.validate_config()

    simulator = experiment.get_simulator_class()
    if not simulator.validate_parameters(
        f.reproduction.mutator.neuron_parameters,
        f.reproduction.mutator.synapse_parameters,
    ):
        warnings.warn(
            "The parameter values for synapses and neurons "
            f"are not valid for the given simulator ({simulator})"
        )

    # start from loaded stats
    if args.load_stats is not None and os.path.isfile(args.load_stats):
        stats = Stats.from_file(args.load_stats)
    else:
        stats = None

    if args.evolution:
        stats = f.evolution(stats)

    if args.save_plot:
        os.makedirs(args.save_plot, exist_ok=True)
        print("Saving plots to: " + os.path.abspath(args.save_plot))

    if args.best:
        best_network, best_fitness = stats.get_best_network_alltime()
        fig = best_network.plot_graph(spring_layout_seed=1)
        if args.save_plot:
            tikzplotlib.save(os.path.join(args.save_plot, "best.tex"))
            fig.savefig(
                os.path.join(args.save_plot, "best.pgf"), bbox_inches="tight"
            )
            fig.savefig(
                os.path.join(args.save_plot, "best.png"), bbox_inches="tight"
            )
        elif args.plot:
            fig.show()
        print(f"Best performance: {experiment.fitness([best_network])}")

    if args.test:
        if not isinstance(experiment, Classification):
            print("This experiment does not support test functionality")
        else:
            best_network, _ = stats.get_best_network_alltime()

            test_score = experiment.test_network(best_network, args.test_runs)
            print("Train size: {}".format(len(experiment.y_train)))
            print("Test score: {}".format(test_score))

    fig = stats.plot_progress()
    if args.save_plot:
        tikzplotlib.save(os.path.join(args.save_plot, "progress.tex"))
        fig.savefig(
            os.path.join(args.save_plot, "progress.png"), bbox_inches="tight"
        )
    elif args.plot:
        fig.show()

    if args.stats:
        stats.to_file(args.stats, indent=None)
