# Evolutionary Optimization of Spiking Neural Networks

This repository contains a framework for conducting experiments regarding evolutionary optimization in spiking neural networks.
In the following, we briefly explain how to run experiments and use the configuration.

## Setup

To run the framework, we describe the system setup via Conda and Docker.
We need a python environment with the packages specified in `environment.yml`.
At the moment, we target Python 3.8.

### Conda

For the conda environment, the required packages are specified in the `environment.yml`.
To create the environment, use the following command:

```bash
conda env create -f environment.yml
```

After the creation, the environment needs to be activated via:

```bash
conda activate master
```

### Docker

Alternatively, the framework can be executed within a Docker container.
Therefore, we first build an image via:

```bash
docker build -t master .
```

To run commands inside the container, and execute the framework use:

```bash
docker run -it --rm -v ${PWD}:/project master /bin/bash
```

Once the terminal is set up, the conda environment needs to be activated.

```bash
conda activate master
```

## Run

The main two scripts to start an evolutionary optimization are `run_experiment.py` and `grid_search.py`.
Both scripts give more information with the `--help` option.
The configuration for an experiment is given in YAML format.
For both scripts a base Configuration as in [Configuration](#configuration) is required, this is provided via the `--config` option.
For the grid search, an additional [Grid Configuration](#grid-configuration) is required.

## Configuration

To create the default configuration, use the script `scripts/generate_default_config.py`. Note: the working directory has to be the scripts directory to work.

To generate the default configuration as `script/config.yaml` use:

```bash
cd scripts/
python generate_default_config.py --file config.yaml
```

### Default configuration
The current default configuration is:
````yaml
experiment: # Select experiment to run
experiment_options: {} # Select experiment to run
number_of_mutations: # Amount of mutations to apply, for a single network
  type: fixed
  value: 7
neuron_parameters: # Which parameters a neuron has
  threshold:
    type: random_int
    min: 0
    max: 127
  leak:
    type: random_choice
    values:
    - 1
    - 5
    - 10
    - 20
    - 40
synapse_parameters: # Which parameters a synapse has
  weight:
    type: random_int
    min: 0
    max: 127
  delay:
    type: random_int
    min: 0
    max: 15
  exciting:
    type: random_bool
mutation_rates: # Probability of mutations to happen
  add_node: 0.08
  delete_node: 0.08
  add_edge: 0.15
  delete_edge: 0.15
  node_param: 0.27
  edge_param: 0.27
generate_hidden_neurons:
  type: random_int
  min: 0
  max: 2
generate_synapses:
  type: random_int
  min: 0
  max: 2
reproduction_rates: # Probabilities to apply mutation/crossover/merge
  mutation: 0.85
  crossover: 0.1
  merge: 0.05
selection_type: tournament
selection_arguments: {} # set custom arguments for the selection, e.g. k and p for tournament_selection
random_factor: 0.1 # introduce new networks into the population in each epoch in percent
num_best: 2 # keep this amount of best networks for new population
population_size: 500 # amount of networks in each population
num_generations: 50 # amount of epochs to simulate
fitness_target: # Finish simulation early, if target is reached
print_status: true # Print progress of the evolution
save_stat_regularly: false # Save the stats after each epoch
cache_evolution: true # Whether to reevalute existing networks during evolution
cache_evolution_warm_up: true # Whether to include evaluated stats into the cache
seed: # Seed for all network related random operations

````

### Configuration Options

- `experiment` Possible values for the experiment option, `experiment_options` are given as subpoints
  - `xor` XOR experiment with Brian
    - `poisson` boolean (Default: False),  whether to use Poisson encoding
    - `rounds` integer (Default: 1), how often to evaluate each sample
    - `decoder_type` {"binary", "classification"} (Default: "classification"), which decoder to use
    - `binary_boundary` integer (Default: 75), boundary to use for binary decoder
  - `cart_pole` Cart Pole Balancing control task
    - `samples_per_network` integer (Default: 10), Number of evaluations during training
    - `poisson` boolean (Default: True), Whether to use Poisson encoding for the observation input spikes
  - `classification` Several classifications tasks
    - `task` {"iris", "wine", "breast"} (Default: "iris"), data set for the evaluation
    - `train_size` integer or float (Default: 0.8), size of the training set either specifically, or in percent
    - `rounds` int (Default: 1), number of rounds to evaluate samples during training on each network
    - `split_seed` int (Default: 1), the seed for the random splitting of training and test data
    - `poisson` boolean (Default: True), whether to use Poisson encoding for encoding input data
    - `penalize_network_size` boolean (Default: True), whether to include a penalty for network size in fitness evaluation
  - `dummy` an experiment, to check the functioning of the evolution, without simulation, the fitness function is the number of hidden neurons + synapses
- `selection_type` {"tournament"} (Default: "tournament"), currently only tournament selection is supported
  - `k` (Default: 10) and `p` (Default: 1) are `selection_arguments` for tournament selection
- `neuron_parameters` Parameters for neurons, key identifies the name, value is given as random parameter configuration, supported parameters are given in [Brian Parameters](#brian-parameters)
- `synapse_parameters` Same as `neuron_parameters`

Additional options in the default configuration are directly explained via comments.

#### Random Parameter Configuration

For some parameters, several options for random generation are available. These values are identified by the `type` field.
There are the following types implemented:

- `fixed` Static value, is not considered for mutable parameters (for example in `neuron_parameters`)
  - `value` Always the given value is chosen
- `random_int` Random integer number within (and including) the bounds
  - `min` integer, lower bound
  - `max` integer, upper bound
- `random_choice` Random value from the given set, each having the same probability (1/n)
  - `values` list of different values
- `random_bool` Random boolean value (True/False), no further configuration
- `random_rates` Random value chosen from a dictionary
  - Example: The keys represent the values to choose from, and the value represents the probability. Here, True has a probability of 1/(1+2)=1/3, and False a probability of 2/(1+3) = 2/3
    - True: 1
    - False: 2

#### Brian Parameters

At the moment, we support the following properties for neurons and synapses in the Brian simulator.
While additional properties could be defined in the configuration, these won't have any impact on the simulation.

Neurons:

- `threshold` integer, value once a neuron spikes; should be a maximum of 127
- `leak` integer (Default: 10), leakage parameter

Synapses:

- `weight` integer, weight to modify potential of the postsynaptic neuron
- `delay` integer, delay spikes by the specified amount in ms
- `exciting` boolean, whether it is an inhibitory or excitatory synapse

### Grid Configuration

For a grid search multiple variations can be compared. Therefore,  we support two levels of configuration, to compare variants.
Additionally, a base configuration is required.
The idea is, that some values are overwritten for the base configuration.

- `options` List, with all options to compare
  - `alternatives` each option should be identified by a dict with the key
    - a list of alternatives is specified
- `pool_size` number of threads for simultaneous evaluation. If not specified, all available cores are used.
- `save` relative path to a directory, where all output information is saved to

#### Grid Configuration Example

The following grid configuration executes 4 evolutionary optimizations.
The alternatives of each option are compared once.
Therefore, here we execute one experiment with a population size of 500 and 10 generations on the xor experiment once, and on the iris experiment once.
Additionally, both experiments are run with 20 generations and a population size of 250.
The results are saved to the directory `computations/example`

```yaml
save: computations/example
options:
  - alternatives:
      - num_generations: 10
        population_size: 500
      - num_generations: 20
        population_size: 250
  - alternatives:
      - experiment: iris
      - experiment: xor
```

## Directories and Files

- `experiment` Contains files regarding the different tasks, we implemented to conduct experiments.
  - `brian` Experiments to use in conjunction with the Brian simulator
- `network` On this level contains the implementation of neurons, synapses and a neuronal network
  - `decoder`, `encoder` Interfaces for different encoders/decoders
    - `brian` Implementations for the Brian simulator
    - `lava` Prototypes for the lava framework
  - `evolution` Implementation of the evolutionary algorithm, including an interface to evaluate the computations (`stats.py`)
- `scripts` Additional scripts
  `generate_default_config.py` Allows generating a default configuration
- `simulator` Implementation of simulators as backend
  - `brian.py` Conversion of and execution of our networks in Brian
  - `lava.py` Prototype for converting networks for Lava
  - `grid.py` Metaclass to execute a hyperparameter search
  - `simulator.py` Interface for a simulator
- `test` Tests of the framework
- `utility` Additional functions and classes, that are used framework wide and do not belong to any of the above categories.
- `run_experiment.py` Run a single neuroevolutionary experiment
- `grid_search.py` Perform a hyperparameter grid search on neuroevolutionary experiments

## Development

In order to run code-style checks before each commit, we use pre-commit.
It can be activated via:

```bash
pre-commit install
```

To run all hooks without a git commit, you can also use:

```bash
pre-commit run --all-files
```

### Test

We also included tests for the framework. These will run automatically before each commit via pre-commit.
They can also be manually executed via:

```bash
python -m unittest discover
```
