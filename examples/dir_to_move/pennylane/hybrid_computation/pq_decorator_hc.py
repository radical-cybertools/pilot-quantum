import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np

import pilot.pilot_compute_service as pcs

plt.set_loglevel("warning")

import os

import matplotlib.pyplot as plt

RESOURCE_URL_HPC = "ssh://localhost"
WORKING_DIRECTORY = os.path.join(os.environ["HOME"], "work")

pilot_compute_description_dask = {
    "resource": RESOURCE_URL_HPC,
    "working_directory": WORKING_DIRECTORY,
    "queue": "debug",
    "type": "dask",
    "walltime": 15,
    "project": "m4408",
    "number_of_nodes": 1,
    "cores_per_node": 2,
    "scheduler_script_commands": ["#SBATCH --constraint=cpu"]
}


def start_pilot():
    p = pcs.PilotComputeService()
    dp = p.create_pilot(pilot_compute_description=pilot_compute_description_dask)
    dp.wait()
    return dp


dask_pilot = start_pilot()
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


def get_init_params(init_params):
    return np.array(init_params, requires_grad=True)


@dask_pilot.task
def get_optimizer():
    return qml.GradientDescentOptimizer(stepsize=0.4)


@dask_pilot.task
def training(opt, init_params, cost, steps):
    params = init_params
    training_steps, cost_steps = [], []  # to record the costs as training progresses
    for i in range(steps):
        params = opt.step(cost, params)
        training_steps.append(i)
        cost_steps.append(cost(params))
    return params, training_steps, cost_steps


if __name__ == "__main__":
    try:
        # Workflow
        init_params = [0.01, 0.01]
        steps = 50

        opt = get_optimizer()
        params = get_init_params(init_params)
        t = training(opt, params, cost, steps)
        opt_params, training_steps, cost_steps = t.result()

        # Plot the results
        fig, ax = plt.subplots(1, 1, figsize=(7, 5), facecolor="w")
        ax.plot(training_steps, cost_steps)
        ax.set_xlabel("Training steps")
        ax.set_ylabel("Cost")
        ax.set_title("Cost vs. Training steps")
        plt.tight_layout()
        plt.show()
    finally:
        if dask_pilot:
            dask_pilot.cancel()
