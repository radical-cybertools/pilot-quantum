import os
import socket
import time
import distributed
import pennylane as qml

import pilot.pilot_compute_service
import ray

RESOURCE_URL_HPC = "slurm://localhost"
WORKING_DIRECTORY = os.path.join(os.environ["PSCRATCH"], "work")

pilot_compute_description_dask = {
    "resource": RESOURCE_URL_HPC,
    "working_directory": WORKING_DIRECTORY,
    "number_cores": 1,
    "queue": "normal",
    "walltime": 5,
    "type": "ray",
    "project": "m4408",
    "os_ssh_keyfile": "~/.ssh/nersc",
    "scheduler_script_commands": ["#SBATCH --constraint=cpu"]
}

@ray.remote
def run_circuit():
    start = time.time()
    wires = 4
    layers = 1

    dev = qml.device('default.qubit', wires=wires, shots=None)

    @qml.qnode(dev)
    def circuit(parameters):
        qml.StronglyEntanglingLayers(weights=parameters, wires=range(wires))
        return [qml.expval(qml.PauliZ(i)) for i in range(wires)]

    shape = qml.StronglyEntanglingLayers.shape(n_layers=layers, n_wires=wires)
    weights = qml.numpy.random.random(size=shape)
    val = circuit(weights)
    end=time.time()
    return (end-start)

@ray.remote
def f(x):
    return x * x    


if __name__ == "__main__":
    ray_pilot = pilot.pilot_compute_service.PilotComputeService.create_pilot(pilot_compute_description_dask)
    print("waiting for dask pilot to start")
    ray_pilot.wait()
    print("waiting done for dask pilot to start")
    ray_client = ray_pilot.get_context()
    #ray_client.cluster_resources()

    # call the remote function ten times in parallel
    # and return the results as a list
    with ray_client:
        print(ray.get([f.remote(i) for i in range(10)]))
        print(ray.get([run_circuit.remote() for i in range(10)]))
    
    ray_pilot.cancel()