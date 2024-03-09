import os
import sys
import time

import pennylane as qml

from pilot.pilot_compute_service import PilotComputeService

sys.path.insert(0, os.path.abspath('../..'))

RESOURCE_URL_HPC = "slurm://localhost"
WORKING_DIRECTORY = os.path.join(os.environ["HOME"], "work")

pilot_compute_description_dask = {
    "resource": RESOURCE_URL_HPC,
    "working_directory": WORKING_DIRECTORY,
    "queue": "regular",
    "type": "dask",
    "walltime": 5,
    "project": "m4408",
    "number_of_nodes": 2,
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
        dask_client = dask_pilot.get_client()
        print(dask_client.scheduler_info())

        # Execute Classical tasks
        print(dask_client.gather(dask_client.map(square, range(10))))

        # Execute Quantum tasks
        print(dask_client.gather(dask_client.map(lambda a: pennylane_quantum_circuit(), range(10))))
        if dask_client:
            dask_client.close()
            time.sleep(2)
    except Exception as ex:
        if dask_pilot:
            dask_pilot.cancel()


