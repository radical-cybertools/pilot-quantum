import os
print("PYTHONPATH:", os.getenv('PYTHONPATH'))
import sys
import socket
import time
import pennylane as qml
from pennylane import numpy as np
from scipy.optimize import minimize

import ray
import pilot
from pilot.pilot_compute_service import PilotComputeService

RESOURCE_URL_HPC = "slurm://localhost"
#WORKING_DIRECTORY = os.path.join(os.environ["HOME"], "work")
WORKING_DIRECTORY = os.path.join(os.environ["PSCRATCH"], "work")

pilot_compute_description_ray_cpu = {
    "resource": RESOURCE_URL_HPC,
    "working_directory": WORKING_DIRECTORY,
    "number_of_nodes": 1,
    "cores_per_node": 24,
    "gpus_per_node": 0,
    "queue": "debug",
    "walltime": 30,
    "type": "ray",
    "project": "m4408",
    "conda_environment": "/pscratch/sd/l/luckow/conda/pilot-quantum",
    "scheduler_script_commands": ["#SBATCH --constraint=cpu"]
}

pilot_compute_description_ray_gpu = {
    "resource": RESOURCE_URL_HPC,
    "working_directory": WORKING_DIRECTORY,
    "number_of_nodes": 1,
    "cores_per_node": 24,
    "gpus_per_node": 4,
    #"queue": "shared_interactive",
    "queue": "debug",
    "walltime": 30,
    "type": "ray",
    "project": "m4408",
    "conda_environment": "/pscratch/sd/l/luckow/conda/pilot-quantum",
    "scheduler_script_commands": ["#SBATCH --constraint=gpu"]
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
def fit_circuit_to_target():
    # Add to requirements.txt: jax, scipy
    start = time.time()    

    wires = 11
    layers = 2
    dev = qml.device('default.qubit', wires=wires)
    
    @qml.qnode(dev)
    def circuit(parameters):
        parameters = parameters.reshape((layers, wires, 3))
        qml.StronglyEntanglingLayers(weights=parameters, wires=range(wires))
        return qml.state()

    def loss_fn(params, target):
        return 1 - np.sum(np.abs(target.conj() * circuit(params)))

    grad_fn = qml.grad(loss_fn, argnum=0)
    weights = qml.numpy.random.random(size=layers*wires*3).flatten()

    # Generate target state, this would be replaced by the target state that reprents the image
    target = np.random.random([2 ** wires]) + 1j * np.random.random([2 ** wires])
    target /= np.linalg.norm(target)
    target = np.array(target, requires_grad=False)

    losses = [float(loss_fn(weights, target))]
    def callback(x):
        current_loss = float(loss_fn(x, target))
        losses.append(current_loss)

    # Optimize
    res = minimize(
        fun=lambda param: loss_fn(param, target),
        callback=callback,
        method='BFGS',
        x0=weights,
        options={"return_all": True, 'maxiter': 100, "xrtol": 0, "gtol": 0},
        jac=lambda param: grad_fn(param, target)
        )

    end=time.time()
    return (end-start)

@ray.remote
def f(x):
    return x * x    


def start_pilot(pilot_compute_description):
    pcs = PilotComputeService()
    dp = pcs.create_pilot(pilot_compute_description=pilot_compute_description)
    print("waiting for Ray pilot to start")
    dp.wait()
    return dp


if __name__ == "__main__":

    ray_pilot, ray_client = None, None

    try:
        # Start Pilot
        ray_pilot = start_pilot(pilot_compute_description_ray_cpu)
        
        # Get Dask client details
        ray_client = ray_pilot.get_client()
        
        with ray_client:
            print(ray.get([fit_circuit_to_target.remote() for i in range(10)]))
            print(ray.get([f.remote(i) for i in range(10)]))
            print(ray.get([run_circuit.remote() for i in range(10)]))
    
        ray_pilot.cancel()
    except Exception as e:
        print("An error occurred: %s" % (str(e)))
    finally:
        if ray_pilot:
            ray_pilot.cancel()


 