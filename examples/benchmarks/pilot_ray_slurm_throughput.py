import os

from pilot.pilot_enums_exceptions import ExecutionEngine
import time

from pilot.pilot_compute_service import PilotComputeService
import csv

RESOURCE_URL_HPC = "slurm://localhost"
WORKING_DIRECTORY = os.path.join(os.environ["HOME"], "work")

pilot_compute_description_ray_cpu = {
    "resource": RESOURCE_URL_HPC,
    "working_directory": WORKING_DIRECTORY,
    "number_of_nodes": 4,
    "cores_per_node": 64,
    "gpus_per_node": 0,
    "queue": "premium",
    "walltime": 30,
    "type": "ray",
    "project": "m4408",
    "scheduler_script_commands": ["#SBATCH --constraint=cpu"]
}

def start_pilot():
    pcs = PilotComputeService(execution_engine=ExecutionEngine.RAY, working_directory=WORKING_DIRECTORY)
    pilot=pcs.create_pilot(pilot_compute_description=pilot_compute_description_ray_cpu)
    pilot.wait()
    return pcs

def sleep(timeSecs):
    time.sleep(timeSecs)

if __name__ == "__main__":
    pcs = None
    try:
        # Start Pilot
        pcs = start_pilot()
        
        with open("result_benchmark.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["cores", "tasks", "runtime_secs", "throughput"])        
            for t in [1024,2048,4096,8192]:
                for m in range(3):
                    start_time = time.time()
                    tasks = []
                    for i in range(t):
                        k = pcs.submit_task(sleep, 0, resources={'num_cpus': 1, 'num_gpus': 0, 'memory': None})
                        tasks.append(k)                                
                    pcs.wait_tasks(tasks)
                    end_time = time.time()
                    throughput = t / (end_time - start_time)
                    writer.writerow([pilot_compute_description_ray_cpu["number_of_nodes"]*pilot_compute_description_ray_cpu["cores_per_node"], t, (end_time - start_time), throughput])
                    file.flush()
        
        
    finally:
        if pcs:
            pcs.cancel()

 