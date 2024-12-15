import os
import time

from pilot.pilot_compute_service import ExecutionEngine, PilotComputeService
from time import sleep

RESOURCE_URL_HPC = "ssh://localhost"
WORKING_DIRECTORY = os.path.join(os.environ["HOME"], "work")

pilot_compute_description_ray = {
    "resource": RESOURCE_URL_HPC,
    "working_directory": WORKING_DIRECTORY,
    "type": "ray",
    "number_of_nodes": 1,
    "cores_per_node": 10,
}


def start_pilot():
    pcs = PilotComputeService(execution_engine=ExecutionEngine.RAY, working_directory=WORKING_DIRECTORY)
    pcs.create_pilot(pilot_compute_description=pilot_compute_description_ray)
    return pcs

def sleep(timeSecs):
    time.sleep(timeSecs)

if __name__ == "__main__":
    pcs = None
    try:
        # Start Pilot
        pcs = start_pilot()        
        for i in range(2):
            tasks = []
            start_time = time.time()
            for i in range(10):            
                k = pcs.submit_task(sleep, 0, resources={'num_cpus': 1, 'num_gpus': 0, 'memory': None})
                tasks.append(k)                    
            pcs.wait_tasks(tasks)
            end_time = time.time()
            print(f"Execution time: {end_time-start_time}")
    finally:
        if pcs:
            pcs.cancel()