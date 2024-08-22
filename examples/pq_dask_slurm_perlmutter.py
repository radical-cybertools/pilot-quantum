import os

import pennylane as qml
from pilot.pilot_compute_service import PilotComputeService
from time import sleep

RESOURCE_URL_HPC = "slurm://localhost"
WORKING_DIRECTORY = os.path.join(os.environ["HOME"], "work")

pilot_compute_description_dask = {
    "resource": RESOURCE_URL_HPC,
    "working_directory": WORKING_DIRECTORY,
    "type": "dask",
    "number_of_nodes": 1,
    "cores_per_node": 2,       
    #"queue": "shared_interactive",
    "queue": "debug",
    "walltime": 30,
    "project": "m4408",
    "conda_environment": "/pscratch/sd/l/luckow/conda/pilot-quantum",
    "scheduler_script_commands": ["#SBATCH --constraint=gpu"]
}


def start_pilot():
    pcs = PilotComputeService(working_directory=WORKING_DIRECTORY)
    dp = pcs.create_pilot(pilot_compute_description=pilot_compute_description_dask)
    dp.wait()
    return dp

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

        print("Start sleep 1 tasks")
        tasks = []
        for i in range(10):
            k = dask_pilot.submit_task(sleep, 1, task_name=f"task_sleep-{i}")
            tasks.append(k)

        dask_pilot.wait_tasks(tasks)
        print("Start Pennylane tasks")
        tasks = []
        for i in range(10):
            k = dask_pilot.submit_task(pennylane_quantum_circuit, task_name=f"task_pennylane-{i}")
            tasks.append(k)

        dask_pilot.wait_tasks(tasks)        
    finally:
        if dask_pilot:
            dask_pilot.cancel()
