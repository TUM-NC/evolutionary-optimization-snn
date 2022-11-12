import random
from typing import List, Optional

import gym
from gym.envs.classic_control import CartPoleEnv

from experiment.brian.brian_experiment import BrianExperiment
from network.decoder.brian.classification import ClassificationBrianDecoder
from network.encoder.brian.float import FloatBrianEncoder
from network.network import Network
from utility.list_operation import remove_multiple_indices
from utility.visualisation import render_frames_as_animation


class CartPoleBalancing(BrianExperiment):
    encoder: FloatBrianEncoder
    decoder: ClassificationBrianDecoder

    samples_per_network: int

    random_generator: random.Random

    def __init__(self, samples_per_network=10, poisson=True):
        self.samples_per_network = samples_per_network
        self.encoder = FloatBrianEncoder(number_of_neurons=4, poisson=poisson)
        self.decoder = ClassificationBrianDecoder(classes=2)

        self.random_generator = random.Random()

    def _init_envs(self, networks_for_simulation: List[Network]):
        """
        Init one environment for each network

        :param networks_for_simulation:
        :return: environments and inputs
        """
        # initialize one env for each network and sample
        inputs = []
        envs = []
        for _ in networks_for_simulation:
            env: CartPoleEnv = gym.make("CartPole-v1")
            env.action_space.seed(self.random_generator.randint(0, 1000000))
            env.seed(self.random_generator.randint(0, 1000000))
            env.reset()
            envs.append(env)

            # add initial state values to inputs
            values = self.convert_state_to_norm(env.state)
            inputs.append(values)
        return envs, inputs

    def _apply_output(self, envs, outputs):
        """
        Apply the given outputs to the envs

        :param envs:
        :param outputs:
        :return: returns the next inputs and the finished simulations
        """
        inputs = []
        finished_simulations = set()

        # apply next action
        for index, env in enumerate(envs):
            action, _ = outputs[index]
            # for now: when no action can be chosen, chose random
            if action == -1:
                action = env.action_space.sample()

            next_state, reward, done, info = env.step(action)
            inputs.append(self.convert_state_to_norm(next_state))

            if done:
                finished_simulations.add(index)

        return inputs, finished_simulations

    def simulate(self, networks: List[Network]):
        """
        Simulate the cart pole balancing for a set of lists
        Prints out the status of each time step simulation
        :param networks:
        :return:
        """
        # repeat networks for each sample
        simulation_networks = [
            n for n in networks for _ in range(self.samples_per_network)
        ]
        for network in simulation_networks:
            # start with an empty output for each simulation
            self.set_output_by_network(network, [])
        envs, inputs = self._init_envs(simulation_networks)

        line = ""
        for t in range(500):
            to_simulate = len(simulation_networks)
            if to_simulate == 0:
                break
            print(" " * len(line), end="\r")  # clear line before new line
            line = "Time step: {} - networks left: {}".format(t, to_simulate)
            print(line, end="\r")
            # simulate
            simulator = self._get_simulator(simulation_networks, inputs)
            outputs = simulator.simulate()

            inputs, done = self._apply_output(envs, outputs)
            for index in done:
                network = simulation_networks[index]
                rewards = self.get_output_by_network(network)
                rewards.append(t)
                self.set_output_by_network(network, rewards)

            # remove networks from simulation, that are already done
            simulation_networks = remove_multiple_indices(
                simulation_networks, done
            )
            inputs = remove_multiple_indices(inputs, done)
            envs = remove_multiple_indices(envs, done)

    def convert_state_to_norm(self, state):
        """
        Convert the state from gym to values between 0 and 1

        :param state:
        :return:
        """
        cart, cart_velocity, pole_angle, pole_velocity = state

        cart_norm = self.norm_value(cart, 2.4)
        cart_velocity_norm = self.norm_value(cart_velocity, 2)
        pole_angle_norm = self.norm_value(pole_angle, 0.2095)
        pole_velocity_norm = self.norm_value(pole_velocity, 2)

        return (
            cart_norm,
            cart_velocity_norm,
            pole_angle_norm,
            pole_velocity_norm,
        )

    @staticmethod
    def norm_value(value, variation):
        """
        Norm a value based on the range
        Minimum value is 0, maximum value is 1

        :param value:
        :param variation:
        :return:
        """
        limit = max(-variation, min(variation, value))
        return (limit + variation) / (2 * variation)

    def single_fitness(self, network: Network):
        """
        The average of the sample times is the output

        :param network:
        :return:
        """
        calculated = self.get_output_by_network(network)
        average = sum(calculated) / len(calculated)
        return average

    def render_series(self, network: Network, seed=None):
        """
        Render one series of pole balancing for the given network

        :param seed:
        :param network:
        :return:
        """
        simulation_networks = [network]
        envs, inputs = self._init_envs(simulation_networks)
        frames = []

        if seed:
            envs[0].action_space.seed(seed)
            envs[0].seed(seed=seed)
            envs[0].reset()

        for t in range(500):
            print("Timestep {}".format(t), end="\r")

            # TODO: prevent showing window
            frames.append(envs[0].render(mode="rgb_array"))
            simulator = self._get_simulator(simulation_networks, inputs)
            outputs = simulator.simulate()

            inputs, done = self._apply_output(envs, outputs)
            if len(done) > 0:
                break

        return render_frames_as_animation(frames)

    def set_seed(self, seed: Optional[int] = None):
        super().set_seed(seed)
        self.random_generator.seed(seed)
