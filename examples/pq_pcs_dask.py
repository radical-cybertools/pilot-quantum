import os

import pennylane as qml
from pilot.pilot_compute_service import PilotComputeService
from time import sleep

RESOURCE_URL_HPC = "ssh://localhost"
WORKING_DIRECTORY = os.path.join(os.environ["HOME"], "work")

pilot_compute_description_dask = {
    "resource": RESOURCE_URL_HPC,
    "working_directory": WORKING_DIRECTORY,
    "type": "dask",
    "number_of_nodes": 2,
    "cores_per_node": 10,
}


def start_pilot():
    pcs = PilotComputeService(working_directory=WORKING_DIRECTORY)
    pcs.create_pilot(pilot_compute_description=pilot_compute_description_dask)
    return pcs

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
    try:
        # Start Pilot
        pcs = start_pilot()

        tasks = []
        for i in range(10):
            k = pcs.submit_task(sleep, 3)
            tasks.append(k)

        pcs.wait_tasks(tasks)

        tasks = []
        for i in range(10):
            k = pcs.submit_task(pennylane_quantum_circuit, task_name = f"task_pennylane-{i}" )
            tasks.append(k)

        pcs.wait_tasks(tasks)        
    finally:
        if pcs:
            pcs.cancel()
