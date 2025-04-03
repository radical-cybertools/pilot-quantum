import math
import os
import ray
from pilot.pilot_compute_service import ExecutionEngine, PilotComputeService
from time import sleep
import time
import csv

RESOURCE_URL_HPC = "slurm://localhost"
WORKING_DIRECTORY = os.path.join(os.environ["HOME"], "work")


pilot_compute_description_ray = {
    "resource": RESOURCE_URL_HPC,
    "working_directory": WORKING_DIRECTORY,
    "type": "ray",
    "number_of_nodes": 4,
    "cores_per_node": 256,
    "queue": "premium",
    "walltime": 60,
    "project": "m4408",
    "scheduler_script_commands": ["#SBATCH --constraint=cpu"],
    "name": "cpu_pilot"
}


def start_pilot():
    pcs = PilotComputeService(execution_engine=ExecutionEngine.RAY, working_directory=WORKING_DIRECTORY)
    pcs.create_pilot(pilot_compute_description=pilot_compute_description_ray)
    return pcs



def square(num):
    return num ** 2

def sleep(num=0):
    sleep(num)
    return num

if __name__ == "__main__":
    pcs = None
    try:
        # Start Pilot
        start_time = time.time()
        pcs = start_pilot()
        end_time = time.time()
        startup_csv = "pilot_startup_times.csv"
        with open(startup_csv, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if csvfile.tell() == 0:  # Write header if file is empty
                writer.writerow(['Startup Time (s)', 'Nodes', 'Cores per Node', 'Timestamp'])
            startup_time = end_time - start_time
            print(f"Pilot startup time: {startup_time:.2f}s with {pilot_compute_description_ray['number_of_nodes']} nodes and {pilot_compute_description_ray['cores_per_node']} cores per node")
            writer.writerow([
                f"{startup_time:.2f}",
                pilot_compute_description_ray['number_of_nodes'],
                pilot_compute_description_ray['cores_per_node'],
                time.strftime('%Y-%m-%d %H:%M:%S')
            ])
        
        ray_client = pcs.get_client()
        with ray_client:
            print(ray.get([ray.remote(square).remote(i) for i in range(10)]))
            

        # Create/open CSV file for writing results
        csv_filename = f"ray_benchmark_results_{pilot_compute_description_ray['number_of_nodes']}nodes_{pilot_compute_description_ray['cores_per_node']}cores.csv"
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Task Size', 'Duration (s)', 'Throughput (tasks/s)', 'Nodes', 'Cores per Node', 'Timestamp'])

            task_sizes = [128, 256, 512, 1024, 2048, 4096, 8192]
            for size in task_sizes:
                for run in range(3):  # Repeat 3 times
                    tasks = []
                    start_time = time.time()
                    print(f"Run {run+1}: Submitting {size} tasks...")
                    for i in range(size):
                        k = pcs.submit_task(sleep, 0, resources={'num_cpus': 1, 'num_gpus': 0, 'memory': None})
                        tasks.append(k)
                    pcs.wait_tasks(tasks)
                    end_time = time.time()
                    duration = end_time - start_time
                    throughput = size / duration
                    print(f"Run {run+1}: Submitted {size} tasks in {duration:.2f} seconds")
                    print(f"Run {run+1}: Throughput: {throughput:.2f} tasks/second")
                    
                    # Write results to CSV
                    writer.writerow([
                        size, 
                        f"{duration:.2f}", 
                        f"{throughput:.2f}",
                        pilot_compute_description_ray['number_of_nodes'],
                        pilot_compute_description_ray['cores_per_node'],
                        time.strftime('%Y-%m-%d %H:%M:%S')
                    ])

               
                #print(pcs.get_results(tasks))
        
       

       
    finally:
        if pcs:
            pcs.cancel()