from typing import List, Optional, Tuple

import numpy as np
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import InPort
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

from network.decoder.classification import ClassificationDecoder
from network.decoder.lava.decoder import LavaDecoder, OutputProcess


class ClassificationLavaDecoder(ClassificationDecoder, LavaDecoder):
    spikes: Optional[List[int]] = None

    def get_value(self) -> Tuple[int, List[int]]:
        """
        get the value of the spikes
        :return:
        """
        if self.spikes is None:
            raise RuntimeError("Spikes have to be set, before getting a value")

        spikes_per_class = self.spikes
        classification = self.get_best_class(spikes_per_class)

        self.spikes = None

        return classification, spikes_per_class

    def get_output_process(self):
        """
        Return the lava output process for the simulator
        :return:
        """
        return ClassificationOutputProcess(classes=self.number_of_neurons)

    def set_value(self, output_process: "ClassificationOutputProcess"):
        """
        Needs the output process, can ready value from that process
        :param output_process:
        :return:
        """
        self.spikes = (
            output_process.spikes_accum.get().astype(np.int32).tolist()
        )


class ClassificationOutputProcess(OutputProcess):
    def __init__(self, classes):
        super().__init__()
        shape = (classes,)
        self.spikes_in = InPort(shape=shape)
        self.spikes_accum = Var(
            shape=shape
        )  # Accumulated spikes for classification


@implements(proc=ClassificationOutputProcess, protocol=LoihiProtocol)
@requires(CPU)
class PyClassificationOutputProcessModel(PyLoihiProcessModel):
    spikes_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    spikes_accum: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=32)

    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)

    def run_spk(self):
        """Spiking phase: executed unconditionally at every time-step"""
        spk_in = self.spikes_in.recv()
        self.spikes_accum = self.spikes_accum + spk_in
