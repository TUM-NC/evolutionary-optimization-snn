# hack: include parent directory for imports
import csv
import sys

import gym
from gym.envs.classic_control import CartPoleEnv
from sklearn import datasets

from network.network import Network
from network.neuron import Neuron
from network.synapse import Synapse
from utility.visualisation import render_frame

sys.path.append("..")


def plot_sample_network():
    n = Network([Neuron(0), Neuron(1)], [Neuron(2)])
    n.add_neuron(Neuron(3))
    n.add_synapse(Synapse(0, 2))
    n.add_synapse(Synapse(1, 3))
    n.add_synapse(Synapse(3, 2))
    n.add_synapse(Synapse(0, 2))
    n.add_synapse(Synapse(0, 1))
    n.add_synapse(Synapse(3, 0))
    # n.add_synapse(Synapse(2,2)) # visualization buggy, when small figure
    fig = n.plot_graph(spring_layout_seed=5, legend=True, figure_size=(6, 2.5))
    fig.savefig(f"example-network.pgf", bbox_inches="tight")


def plot_cart_pole(intial_rendering=True, further_rendering=True):
    env: CartPoleEnv = gym.make("CartPole-v1")
    env.action_space.seed(42)
    env.seed(seed=42)
    env.reset()

    # initial rendering
    if intial_rendering:
        initial = env.render(mode="rgb_array")
        fig, _ = render_frame(initial, dpi=300)

        fig.savefig("cart-initial.png", bbox_inches="tight")

    #  render a picture from the series
    if further_rendering:
        seed = 98
        env.seed(seed)
        env.action_space.seed(seed)
        env.reset()

        for i in range(500):
            action = env.action_space.sample()
            observation, reward, terminated, truncated = env.step(action)

            if i == 69:
                state_frame = env.render(mode="rgb_array")
                fig, _ = render_frame(state_frame, dpi=300)

                fig.savefig("cart-falling.png", bbox_inches="tight")
                break

            if terminated or truncated:
                break

    env.close()


def save_iris_dataset():
    dataset = datasets.load_iris()
    csv_path = "iris.csv"

    header = dataset.feature_names + ["class"]
    values = [
        list(v) + [target] for v, target in zip(dataset.data, dataset.target)
    ]

    with open(csv_path, "w") as f:
        writer = csv.writer(f, delimiter=",", quotechar='"')
        writer.writerow(header)
        writer.writerows(values)


def main():
    plot_sample_network()
    plot_cart_pole()
    save_iris_dataset()


if __name__ == "__main__":
    main()
