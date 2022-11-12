from typing import Iterable

import matplotlib.pyplot as plt
from brian2 import SpikeMonitor, StateMonitor, ms
from matplotlib import animation

from network.network import Network, NeuronType, get_neuron_color


def render_frames_as_animation(frames):
    """
    Render the given frames as animation

    :param frames:
    :return:
    """
    fig, image = render_frame(frames[0])

    def animate(i):
        image.set_data(frames[i])

    anim = animation.FuncAnimation(
        fig, animate, frames=len(frames), interval=50
    )

    return anim


def render_frame(frame, dpi=72):
    """
    render a single frame from gym rgb

    :param frame:
    :return:
    """
    fig, ax = plt.subplots(
        figsize=(frame.shape[1] / 72.0, frame.shape[0] / 72.0), dpi=300
    )

    image = ax.imshow(frame)
    ax.axis("off")
    return fig, image


def plot_spike_timing(
    spikes: SpikeMonitor,
    network: Network,
    uids: Iterable[int] = None,
    legend=False,
    neuron_colors=True,
):
    """
    Plot the spike timing diagram

    :param spikes:
    :param network:
    :param uids:
    :param legend: whether to show a legend
    :return: figure and ax from matplotlib
    """
    if uids is None:
        uids = network.get_all_neurons_uid()

    fig, ax = plt.subplots()

    brian_neuron_order = network.get_all_neurons_uid()
    spike_trains = spikes.spike_trains()

    spike_plot_data = []

    for uid in uids:
        brian_neuron_id = brian_neuron_order.index(uid)
        spike_train = spike_trains[brian_neuron_id] / ms
        spike_plot_data.append(spike_train)

    if neuron_colors:
        colors = [
            get_neuron_color(network.get_neuron_type(uid=uid)) for uid in uids
        ]
    else:
        colors = "black"
    ax.eventplot(spike_plot_data, colors=colors, linelengths=0.5)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron")

    ax.set_yticks(range(len(uids)))
    ax.set_yticklabels([str(uid) for uid in uids])

    if legend:
        labels = [str(uid) for uid in uids]
        ax.legend(labels)

    return fig, ax


def plot_neuron_potential(
    state_monitor: StateMonitor, network: Network, uids: Iterable[int] = None
):
    """
    Plot the potential  of the given neuron uids in a graph
    :param state_monitor:
    :param network:
    :param uids:
    :return: returns fig and ax for further processing
    """
    if uids is None:
        uids = network.get_all_neurons_uid()

    brian_neuron_order = network.get_all_neurons_uid()

    fig, ax = plt.subplots()

    for uid in uids:
        time = state_monitor.t / ms
        brian_neuron_id = brian_neuron_order.index(uid)
        values = state_monitor.v[brian_neuron_id]
        neuron_type = network.get_neuron_type(uid=uid)

        label = ""
        if neuron_type == NeuronType.Input:
            label = "Input"
        elif neuron_type == NeuronType.Output:
            label = "Output"
        elif neuron_type == NeuronType.Hidden:
            label = "Hidden"

        label += " ({})".format(uid)

        ax.plot(time, values, label=label)
    ax.legend()
    return fig, ax
