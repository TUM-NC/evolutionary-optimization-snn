import argparse

from tabulate import tabulate

from simulator.grid import GridSimulator
from utility.grid_configuration import GridConfiguration

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a grid-search for neuroevolution experiments"
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Provide a yaml file for basic configuration",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-g",
        "--grid",
        help="Provide a yaml file for grid configuration",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    gc = GridConfiguration.from_yaml(
        base_file=args.config, grid_file=args.grid
    )

    amount_iterations = gc.get_amount_alternatives()
    print(f"Starting grid-search with {amount_iterations} variations:")

    grid_simulator = GridSimulator(grid_configuration=gc)
    grid_simulator.iterate()

    print(f"Finished grid-search")

    headers, values = grid_simulator.get_table_values()
    pretty_table = tabulate(values, headers=headers)

    print("\n Results:")
    print(pretty_table)
