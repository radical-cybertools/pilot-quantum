import datetime
import logging
import os
import socket
import sys
import time

import dask.bag as db
# Python Dask and Data stack
import distributed

# Pilot-Quantum
import pilot.streaming

logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, os.path.abspath('../../..'))

# Qiskit
from qiskit_aer.primitives import Estimator as AirEstimator
from qiskit_benchmark import generate_data

logging.getLogger('qiskit').setLevel(logging.INFO)
logging.getLogger('qiskit.transpiler').setLevel(logging.WARN)
logging.getLogger('stevedore.extension').setLevel(logging.INFO)

RESOURCE_URL_HPC = "slurm://localhost"
WORKING_DIRECTORY = os.path.join(os.environ["PSCRATCH"], "work")

num_nodes = int(sys.argv[1])
wall_time = int(sys.argv[2])
num_gpus = int(num_nodes) * 4


def start_pilot():
    pilot_compute_description_dask = {
        "resource": RESOURCE_URL_HPC,
        "working_directory": WORKING_DIRECTORY,
        "number_of_nodes": num_nodes,
        "queue": "debug",
        "walltime": wall_time,
        "type": "dask-cuda",
        "project": "m4408",
        "os_ssh_keyfile": "~/.ssh/nersc",
        "scheduler_script_commands": ["#SBATCH --constraint=gpu", f"#SBATCH --gpus={num_gpus}", "#SBATCH --gpu-bind=none"]
    }
    dask_pilot = pilot.streaming.PilotComputeService.create_pilot(pilot_compute_description_dask)
    dask_pilot.wait()
    dask_pilot.get_details()
    dask_client = distributed.Client(dask_pilot.get_details()['master_url'])
    return dask_pilot, dask_client


if __name__ == "__main__":
    num_qubits_arr = [25]
    n_entries = 1024
    results = []
    run_timestamp = datetime.datetime.now()
    RESULT_FILE = "pilot-quantum-summary-gpus" + run_timestamp.strftime("%Y%m%d-%H%M%S") + ".csv"
    dask_pilot = None

    pilot_start_time = time.time()
    dask_pilot, dask_client = start_pilot()
    pilot_end_time = time.time()

    print(dask_client.gather(dask_client.map(lambda a: a * a, range(10))))
    print(dask_client.gather(dask_client.map(lambda a: socket.gethostname(), range(10))))

    for i in range(5):
        for num_qubits in num_qubits_arr:
            classical_compute_start = time.time()
            start_compute = time.time()
            circuits, observables = generate_data(
                depth_of_recursion=1,  # number of circuits and observables
                num_qubits=num_qubits,
                n_entries=n_entries,  # number of circuits and observables => same as depth_of_recursion
                circuit_depth=1,
                size_of_observable=1
            )
            circuits_observables = zip(circuits, observables)
            circuit_bag = db.from_sequence(circuits_observables)
            classical_compute_end = time.time()
            options = {"method": "statevector", "device": 'GPU', "cuStateVec_enable": True, "shots": None}
            print(options)
            estimator = AirEstimator(backend_options=options)

            qauntum_compute_start=time.time()
            circuit_bag.map(lambda circ_obs: estimator.run(circ_obs[0], circ_obs[1]).result()).compute(client=dask_client, )
            qauntum_compute_end = time.time()
            end_compute = time.time()
            # write data to file
            result_string = "Pilot-Dask-Overhead: {}, Number Nodes: {}, Number Qubits: {} Number Circuits {} Compute: {}s classical_compute_time: {}s Quantum compute time: {}s".format(
                pilot_end_time - pilot_start_time, num_nodes, num_qubits, n_entries, end_compute - start_compute, classical_compute_end-classical_compute_start, qauntum_compute_end-qauntum_compute_start)
            print(result_string)
            results.append(result_string)
            with open(RESULT_FILE, "w") as f:
                f.write("\n".join(results))
                f.flush()

    time.sleep(30)
    dask_pilot.cancel()
