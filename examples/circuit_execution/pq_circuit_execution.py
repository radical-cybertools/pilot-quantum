import os

import pennylane as qml
from examples.circuit_execution.qiskit_benchmark import generate_data
from pilot.pilot_compute_service import PilotComputeService
from time import sleep
from qiskit_aer.primitives import Estimator as AirEstimator

RESOURCE_URL_HPC = "ssh://localhost"
WORKING_DIRECTORY = os.path.join(os.environ["HOME"], "work")

pilot_compute_description_dask = {
    "resource": RESOURCE_URL_HPC,
    "working_directory": WORKING_DIRECTORY,
    "type": "dask",
    "number_of_nodes": 1,
    "cores_per_node": 10,
}


def start_pilot():
    pcs = PilotComputeService()
    dp = pcs.create_pilot(pilot_compute_description=pilot_compute_description_dask)
    dp.wait()
    return dp

def run_circuit(circ_obs, qiskit_backend_options):
    estimator_result = AirEstimator(backend_options=qiskit_backend_options).run(circ_obs[0], circ_obs[1]).result()
    print(estimator_result)
    return estimator_result

if __name__ == "__main__":
    dask_pilot, dask_client = None, None

    try:
        # Start Pilot
        dask_pilot = start_pilot()

        # Get Dask client details
        dask_client = dask_pilot.get_client()
        print(dask_client.scheduler_info())


        circuits, observables = generate_data(
            depth_of_recursion=1,
            num_qubits=25,
            n_entries=10,
            circuit_depth=1,
            size_of_observable=1
        )

        circuits_observables = zip(circuits, observables)

        tasks = []
        for i, co in enumerate(circuits_observables):
            k = dask_pilot.submit_task(f"task_ce-{i}",run_circuit, co, {"method": "statevector"})
            tasks.append(k)

        dask_pilot.wait_tasks(tasks)
    finally:
        if dask_pilot:
            dask_pilot.cancel()
