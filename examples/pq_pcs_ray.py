import math
import os

import pennylane as qml
import ray
from pilot.pilot_compute_service import ExecutionEngine, PilotComputeService
from time import sleep

RESOURCE_URL_HPC = "ssh://localhost"
WORKING_DIRECTORY = os.path.join(os.environ["HOME"], "work")

pilot_compute_description_ray = {
    "resource": RESOURCE_URL_HPC,
    "working_directory": WORKING_DIRECTORY,
    "type": "ray",
    "number_of_nodes": 2,
    "cores_per_node": 1,
}


def start_pilot():
    pcs = PilotComputeService(execution_engine=ExecutionEngine.RAY, working_directory=WORKING_DIRECTORY)
    pcs.create_pilot(pilot_compute_description=pilot_compute_description_ray)
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

def square(num):
    return num ** 2

if __name__ == "__main__":
    pcs = None
    try:
        # Start Pilot
        pcs = start_pilot()
        
        ray_client = pcs.get_client()
        with ray_client:
            print(ray.get([ray.remote(square).remote(i) for i in range(10)]))
            

        tasks = []
        for i in range(10):
            k = pcs.submit_task(square, i, resources={'num_cpus': 1, 'num_gpus': 0, 'memory': None})
            tasks.append(k)
                    

        pcs.wait_tasks(tasks)
        print(pcs.get_results(tasks))
        
        tasks = []
        
        for i in range(10):
            k = pcs.submit_task(pennylane_quantum_circuit, task_name = f"task_pennylane-{i}" )
            tasks.append(k)

        pcs.wait_tasks(tasks)        
    finally:
        if pcs:
            pcs.cancel()