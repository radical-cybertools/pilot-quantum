import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

from pilot.pilot_compute_service import PilotComputeService

import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

RESOURCE_URL_HPC = "slurm://localhost"
WORKING_DIRECTORY = os.path.join(os.environ["HOME"], "work")

pilot_compute_description_dask = {
    "resource": RESOURCE_URL_HPC,
    "working_directory": WORKING_DIRECTORY,
    "queue": "debug",
    "type": "dask",
    "walltime": 15,
    "project": "m4408",
    "number_of_nodes": 1,
    "cores_per_node": 24,
    "scheduler_script_commands": ["#SBATCH --constraint=cpu"]
}


def start_pilot():
    pcs = PilotComputeService()
    dp = pcs.create_pilot(pilot_compute_description=pilot_compute_description_dask)
    dp.wait()
    return dp




dev_fock = qml.device("strawberryfields.fock", wires=2, cutoff_dim=10)
dev_qubit = qml.device("default.qubit", wires=1)


@qml.qnode(dev_fock)
def photon_redirection(params):
    qml.FockState(1, wires=0)
    qml.Beamsplitter(params[0], params[1], wires=[0, 1])
    return qml.expval(qml.NumberOperator(1))

@qml.qnode(dev_qubit)
def qubit_rotation(phi1, phi2):
    qml.RX(phi1, wires=0)
    qml.RY(phi2, wires=0)
    return qml.expval(qml.PauliZ(0))

def squared_difference(x, y):
    return np.abs(x - y) ** 2


def cost(params, phi1=0.4, phi2=0.8):
    qubit_result = qubit_rotation(phi1, phi2)
    photon_result = photon_redirection(params)
    return squared_difference(qubit_result, photon_result)

def get_optimizer():
    return qml.GradientDescentOptimizer(stepsize=0.4)


def get_init_params(init_params):
    return np.array(init_params, requires_grad=True)

def training(opt, init_params, cost, steps):
    params = init_params
    training_steps, cost_steps = [], []  # to record the costs as training progresses
    for i in range(steps):
        params = opt.step(cost, params)
        training_steps.append(i)
        cost_steps.append(cost(params))
    return params, training_steps, cost_steps


def workflow(init_params, steps):
    opt = get_optimizer()
    params = get_init_params(init_params)
    opt_params, training_steps, cost_steps = training(opt, params, cost, steps)
    return opt_params, training_steps, cost_steps


if __name__ == "__main__":
    dask_pilot, dask_client = None, None

    try:
        # Start Pilot
        dask_pilot = start_pilot()

        # Get Dask client details
        print(dask_pilot.get_details())

        dask_pilot.run_sync_task(workflow, [0.01, 0.01], 50)
    finally:
        if dask_pilot:
            dask_pilot.cancel()