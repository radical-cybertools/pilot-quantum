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
<<<<<<<< HEAD:examples/pq_pcs_dask.py
    "number_of_nodes": 2,
    "cores_per_node": 10,
========
    "number_of_nodes": 1,
    "cores_per_node": 2,
>>>>>>>> main:examples/pq_dask_ssh_localhost.py
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

        print("Start sleep 1 tasks")
        tasks = []
        for i in range(10):
<<<<<<<< HEAD:examples/pq_pcs_dask.py
            k = pcs.submit_task(sleep, 3)
            tasks.append(k)

        pcs.wait_tasks(tasks)

        tasks = []
        for i in range(10):
            k = pcs.submit_task(pennylane_quantum_circuit, task_name = f"task_pennylane-{i}" )
========
            k = dask_pilot.submit_task(f"task_sleep-{i}",sleep, 1)
            tasks.append(k)

        dask_pilot.wait_tasks(tasks)
        print("Start Pennylane tasks")
        tasks = []
        for i in range(10):
            k = dask_pilot.submit_task(f"task_pennylane-{i}", pennylane_quantum_circuit)
>>>>>>>> main:examples/pq_dask_ssh_localhost.py
            tasks.append(k)

        pcs.wait_tasks(tasks)        
    finally:
        if pcs:
            pcs.cancel()
