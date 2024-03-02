import os, time, sys
sys.path.insert(0, os.path.abspath('../../..'))
import socket
import getpass
import datetime

# Python Dask and Data stack
import numpy as np
import pandas as pd
import distributed, dask
from dask.distributed import Client, SSHCluster
import dask.dataframe as dd
import dask.array as da
import dask.bag as db
import hostlist

# Pilot-Quantum
import subprocess
import pilot.streaming

import logging
logging.basicConfig(level=logging.WARNING)

# Qiskit
from qiskit import QuantumCircuit, transpile, execute, Aer
from qiskit_aer import AerSimulator # for GPU
from qiskit.primitives import Estimator
from qiskit_aer.primitives import Estimator as AirEstimator
from qiskit_benchmark import run_graph, generate_data

logging.getLogger('qiskit').setLevel(logging.INFO)
logging.getLogger('qiskit.transpiler').setLevel(logging.WARN)
logging.getLogger('stevedore.extension').setLevel(logging.INFO)





def get_slurm_allocated_nodes():
    print("Init nodefile from SLURM_NODELIST")
    hosts = os.environ.get("SLURM_NODELIST") 
    if hosts == None:
        return ["localhost"]

    print("***** Hosts: " + str(hosts)) 
    hosts=hostlist.expand_hostlist(hosts)
    number_cpus_per_node = 1
    if os.environ.get("SLURM_CPUS_ON_NODE")!=None:
        number_cpus_per_node=int(os.environ.get("SLURM_CPUS_ON_NODE"))
    freenodes = []
    for h in hosts:
        #for i in range(0, number_cpus_per_node):
        freenodes.append((h.strip()))
    return list(set(freenodes))

nodes = get_slurm_allocated_nodes()
num_nodes=len(nodes)
master = nodes[0]


def kill_dask_processes_on_nodes():
    for node in nodes:
        try:
            # SSH into the node and find Dask processes
            ssh_command = f"ssh {node} 'pgrep -f \"dask (scheduler|worker|cuda)\"'"
            process = subprocess.Popen(ssh_command, shell=True, stdout=subprocess.PIPE)
            output, _ = process.communicate()

            # Kill Dask processes found on the node
            if output:
                pids = output.decode().splitlines()
                for pid in pids:
                    ssh_kill_command = f"ssh {node} 'kill -9 {pid}'"
                    subprocess.run(ssh_kill_command, shell=True, check=True)
                logging.debug(f"Dask processes killed on node {node}")
            else:
                logging.debug(f"No Dask processes found on node {node}")
        except subprocess.CalledProcessError as e:
            logging.debug(f"Error executing command on node {node}: {e}")
        except Exception as ex:
            logging.debug(f"An error occurred: {ex}")

def check_dask():
    try:
        import distributed
        client = distributed.Client(master+":8786")
        return client.scheduler_info()
    except Exception as e:
        logging.debug("failed with error: %s" % e)
    return None

def launch_dask_scheduler_via_command_line(scheduler_port=8786):
    logging.debug(f"Launching Dask scheduler started at {master}:{scheduler_port}")
    scheduler_command = f"ssh {master} dask scheduler"
    subprocess.Popen(scheduler_command, shell=True)
    while check_dask() is None:
        time.sleep(1)
    logging.debug(f"Dask scheduler started at {master}:{scheduler_port}")

def launch_dask_workers_via_command_line(scheduler_port=8786):
    master_url = f"{master}:{scheduler_port}"
    logging.debug(f"Launching Workers against Dask scheduler started at {master_url}")
    worker_command = f"dask cuda worker --memory-limit=\"200GB\" {master_url}"

    for node in nodes:
        ssh_worker_command = f"ssh {node} {worker_command} "
        subprocess.Popen(ssh_worker_command, shell=True)
        logging.debug(f"Dask worker started on {node} using {ssh_worker_command}")

def start_dask():
    kill_dask_processes_on_nodes()
    launch_dask_scheduler_via_command_line()
    launch_dask_workers_via_command_line()


def start_pilot():
    start_dask()
    dask_client=Client(f"{master}:8786")        
    return dask_client


if __name__=="__main__":
    num_qubits_arr = [25]
    n_entries = 1024
    results = []
    run_timestamp=datetime.datetime.now()
    RESULT_FILE= "pilot-quantum-gpus-summary-" + run_timestamp.strftime("%Y%m%d-%H%M%S") + ".csv"
    dask_pilot = None
           
    pilot_start_time = time.time()
    dask_client = start_pilot()
    pilot_end_time = time.time()

    print(dask_client.gather(dask_client.map(lambda a: a*a, range(10))))
    print(dask_client.gather(dask_client.map(lambda a: socket.gethostname(), range(10))))

    for i in range(1):
        for num_qubits in num_qubits_arr:
            try:
                classical_compute_start = time.time()
                start_compute = time.time()
                circuits, observables = generate_data(
                    depth_of_recursion=1,  # number of circuits and observables
                    num_qubits=num_qubits,
                    n_entries=n_entries,  # number of circuits and observables => same as depth_of_recursion
                    circuit_depth=1,
                    size_of_observable=1
                )
                #circuits_observables = zip(circuits, observables)
                #circuit_bag = db.from_sequence(circuits_observables)
                classical_compute_end = time.time()
                shots=1024
                options = {"blocking_enable":True, "batched_shots_gpu": True, "blocking_qubits": 25, "executor": dask_client, "max_job_size":1, "method": "statevector", "device": 'GPU', "cuStateVec_enable": True }
                print(options)
                estimator = AirEstimator(backend_options=options)
                qauntum_compute_start=time.time()
                estimator.run(circuits,observables).result()
                #circuit_bag.map(lambda circ_obs: estimator.run(circ_obs[0], circ_obs[1]).result()).compute(client=dask_client)
                qauntum_compute_end = time.time()
                end_compute = time.time()
                # write data to file
                result_string = "Pilot-Dask-Overhead: {}, shots: {}, Number Nodes: {}, Number Qubits: {} Number Circuits {} Compute: {}s classical_compute_time: {}s Quantum compute time: {}s".format(
                    pilot_end_time - pilot_start_time, shots, num_nodes, num_qubits, n_entries, end_compute - start_compute, classical_compute_end-classical_compute_start, qauntum_compute_end-qauntum_compute_start)
                print(result_string)
                results.append(result_string)
                with open(RESULT_FILE, "w") as f:
                    f.write("\n".join(results))
                    f.flush()
            except Exception as e:
                print("Experiment failed. Number Nodes: {} Number Qubits: {} Number Circuits {} Error: {}s".format(num_nodes, num_qubits, n_entries, e))
    
    time.sleep(30)
    # dask_pilot.cancel()