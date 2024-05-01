import os

import pennylane as qml
from pilot.pilot_compute_service import PilotComputeService

RESOURCE_URL_HPC = "ssh://localhost"
WORKING_DIRECTORY = os.path.join(os.environ["HOME"], "work")

pilot_compute_description_dask = {
    "resource": RESOURCE_URL_HPC,
    "working_directory": WORKING_DIRECTORY,
    "queue": "regular",
    "type": "dask",
    "walltime": 5,
    "project": "m4408",
    "number_of_nodes": 1,
    "cores_per_node": 2,
    "scheduler_script_commands": ["#SBATCH --constraint=cpu"]
}


def start_pilot():
    pcs = PilotComputeService()
    dp = pcs.create_pilot(pilot_compute_description=pilot_compute_description_dask)
    dp.wait()
    return dp


def square(a):
    return a * a

def add(a, b):
    return a+b


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
    dask_pilot, dask_client = None, None

    try:
        # Start Pilot
        dask_pilot = start_pilot()

        # Get Dask client details

        a = dask_pilot.submit_task(square, 20)
        b = dask_pilot.submit_task(add, a, 40)
        print(f"{b.result()}\n")

        dask_client = dask_pilot.get_client()
        print(dask_client.scheduler_info())

        # Execute Classical tasks
        print(dask_client.gather(dask_client.map(square, range(10))))

        # Execute Quantum tasks
        print(dask_client.gather(dask_client.map(lambda a: pennylane_quantum_circuit(), range(10))))

    finally:
        if dask_pilot:
            dask_pilot.cancel()
