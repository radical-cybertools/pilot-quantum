import getpass
import os
import sys

import pennylane as qml

from pilot.pilot_compute_service import PilotComputeService

sys.path.insert(0, os.path.abspath('../..'))

RESOURCE_URL_HPC = "ssh://localhost"
WORKING_DIRECTORY = os.path.join(os.environ["HOME"], "work")

pilot_compute_description_dask = {
    "resource": "ssh://{}@localhost".format(getpass.getuser()),
    "working_directory": os.path.join(os.path.expanduser("~"), "work"),
    "type": "dask"
}


def start_pilot():
    pcs = PilotComputeService()
    dp = pcs.create_pilot(pilot_compute_description=pilot_compute_description_dask)
    dp.wait()
    return dp


def square(a):
    return a * a


def pennylane_quantum_circuit():
    wires = 4
    layers = 1
    dev = qml.device('default.qubit', wires=wires, shots=None)

    @qml.qnode(dev)
    def circuit(parameters):
        qml.StronglyEntanglingLayers(weights=parameters, wires=range(wires))
        return [qml.expval(qml.PauliZ(i)) for i in range(wires)]

    shape = qml.StronglyEntanglingLayers.shape(n_layers=layers, n_wires=wires)
    weights = qml.numpy.random.random(size=shape)
    return circuit(weights)


if __name__ == "__main__":
    # Start Pilot
    dask_pilot = start_pilot()

    # Get Dask client details
    dask_client = dask_pilot.get_client()
    print(dask_client.scheduler_info())

    # Execute Classical tasks
    print(dask_client.gather(dask_client.map(square, range(10))))

    # Execute Quantum tasks
    print(dask_client.gather(dask_client.map(lambda a: pennylane_quantum_circuit(), range(10))))
