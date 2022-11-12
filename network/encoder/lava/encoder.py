from lava.magma.core.process.ports.ports import OutPort
from lava.magma.core.process.process import AbstractProcess

from network.encoder.encoder import Encoder


class LavaEncoder(Encoder):
    """
    Abstract class for lava encoders
    """

    def get_input_process(self) -> "InputProcess":
        raise NotImplementedError("Please Implement this method")


class InputProcess(AbstractProcess):
    spikes_out: OutPort
