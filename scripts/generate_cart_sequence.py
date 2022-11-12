import argparse

# hack: include parent directory for imports
import sys

sys.path.append("..")

from experiment.brian.cart_pole_balancing import (  # noqa: E402
    CartPoleBalancing,
)
from network.network import Network  # noqa: E402


def main():
    """
    Allows to generate a configuration yaml to the specified file

    :return:
    """
    parser = argparse.ArgumentParser(description="Generate a configuration")
    parser.add_argument(
        "-n",
        "--network",
        type=str,
        help="Network to load from file",
        required=True,
    )
    parser.add_argument(
        "-s",
        "--save",
        type=str,
        help="Gif to save sequence to",
        default="computations/cart.gif",
    )
    parser.add_argument(
        "--seed", type=int, help="Seed for random generation", default=2
    )
    args = parser.parse_args()

    experiment = CartPoleBalancing(samples_per_network=1)
    experiment.set_seed(args.seed)

    network = Network.from_file(args.network)
    anim = experiment.render_series(network, seed=args.seed)
    anim.save(args.save)


if __name__ == "__main__":
    main()
