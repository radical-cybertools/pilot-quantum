import os
import socket
import time
import pennylane as qml
import pilot.pilot_compute_service
import ray
from pilot.pilot_compute_service import PilotComputeService

RESOURCE_URL_HPC = "slurm://localhost"
WORKING_DIRECTORY = os.path.join(os.environ["PSCRATCH"], "work")

pilot_compute_description_ray = {
    "resource": RESOURCE_URL_HPC,
    "working_directory": WORKING_DIRECTORY,
    "number_of_nodes": 1,
    "cores_per_node": 24,
    "queue": "regular",
    "walltime": 5,
    "type": "ray",
    "project": "m4408",
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


def start_pilot():
    pcs = PilotComputeService()
    dp = pcs.create_pilot(pilot_compute_description=pilot_compute_description_ray)
    dp.wait()
    return dp


if __name__ == "__main__":


    ray_pilot, ray_client = None, None

    try:
        # Start Pilot
        ray_pilot = start_pilot()

        print("waiting for Ray pilot to start")
        ray_pilot.wait()

        # Get Dask client details
        ray_pilot = ray_pilot.get_context()
        
        with ray_client:
            print(ray.get([f.remote(i) for i in range(10)]))
            print(ray.get([run_circuit.remote() for i in range(10)]))
    
        ray_pilot.cancel()
        
    finally:
        if ray_pilot:
            ray_pilot.cancel()


 