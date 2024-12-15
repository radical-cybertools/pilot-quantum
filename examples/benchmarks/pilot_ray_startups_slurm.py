import os

from pilot.pilot_enums_exceptions import ExecutionEngine
import time

from pilot.pilot_compute_service import PilotComputeService

RESOURCE_URL_HPC = "slurm://localhost"
WORKING_DIRECTORY = os.path.join(os.environ["HOME"], "work")

pilot_compute_description_ray_cpu = {
    "resource": RESOURCE_URL_HPC,
    "working_directory": WORKING_DIRECTORY,
    "number_of_nodes": 2,
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
    pcs.create_pilot(pilot_compute_description=pilot_compute_description_ray_cpu)
    return pcs

def sleep(timeSecs):
    time.sleep(timeSecs)

if __name__ == "__main__":
    pcs = None
    try:
        # Start Pilot
        pcs = start_pilot()
        
        tasks = []
        for i in range(10):
            k = pcs.submit_task(sleep, 1, resources={'num_cpus': 1, 'num_gpus': 0, 'memory': None})
            tasks.append(k)
                    

        pcs.wait_tasks(tasks)
    finally:
        if pcs:
            pcs.cancel()

 