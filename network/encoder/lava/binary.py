import numpy as np
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import OutPort
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

from network.encoder.binary import BinaryEncoder
from network.encoder.lava.encoder import InputProcess, LavaEncoder


class BinaryLavaEncoder(BinaryEncoder, LavaEncoder):
    def get_input_process(self):
        """

        :return:
        """
        return BinaryInput(
            self.number_of_neurons, number_of_neurons=self.number_of_neurons
        )


class BinaryInput(InputProcess):
    def __init__(
        self,
        vth: int,
        number_of_neurons: int,
        # TODO: may add multiple values per network
    ):
        super().__init__()
        shape = (number_of_neurons,)

        self.spikes_out = OutPort(shape=shape)
        self.v = Var(shape=shape, init=0)
        self.vth = Var(shape=(1,), init=vth)


@implements(proc=BinaryInput, protocol=LoihiProtocol)
@requires(CPU)
class PyBinaryInputModel(PyLoihiProcessModel):
    spikes_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
    v: np.ndarray = LavaPyType(np.ndarray, int, precision=32)
    vth: int = LavaPyType(int, int, precision=32)

    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)

    # TODO: could add multiple values with mgmt phase

    def run_spk(self):
        """Spiking phase: executed unconditionally at every time-step"""
        # TODO: implement real spike pattern
        pattern = np.array([2, 1])
        self.v[:] = self.v + pattern
        s_out = self.v > self.vth
        self.v[s_out] = 0  # reset voltage to 0 after a spike
        self.spikes_out.send(s_out)
