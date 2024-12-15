from copy import copy
import csv
import logging
import subprocess
import time
import uuid

from distributed import Future
import ray

from pilot.pilot_enums_exceptions import ExecutionEngine, PilotAPIException
from pilot.pcs_logger import PilotComputeServiceLogger
from pilot.plugins.dask_v2 import cluster as dask_cluster_manager
from pilot.plugins.ray_v2 import cluster as ray_cluster_manager

import os
from dask.distributed import wait
from datetime import datetime
from enum import Enum
import csv
import os
import time
import uuid
from datetime import datetime
import threading


METRICS = {
    'task_id': None,
    'pilot_scheduled': None,
    'submit_time': datetime.now(),
    'wait_time_secs': None, 
    'staging_time_secs': 0,
    'input_staging_data_size_bytes': 0,
    'completion_time': None,            
    'execution_secs': None,
    'status': None,
    'error_msg': None,
}

def run_mpi_task(num_procs, script_path, *args):
    """
    Run an MPI script with the given number of processes and additional arguments.
    """
    cmd = ["srun", "-n", str(num_procs), "python", script_path, *args]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(f"stdout:\n{result.stdout}")
    print(f"stderr:\n{result.stderr}")
    return result.stdout, result.stderr 

SORTED_METRICS_FIELDS = sorted(METRICS.keys())

class PilotComputeBase:
    def __init__(self, execution_engine, working_directory):
        self.execution_engine = execution_engine
        self.pcs_working_directory = working_directory        
        if not os.path.exists(self.pcs_working_directory):            
            os.makedirs(self.pcs_working_directory)

        self.metrics_file_name = os.path.join(self.pcs_working_directory, "metrics.csv")
        self.client = None
        self.logger = PilotComputeServiceLogger(self.pcs_working_directory)
        
        with open(self.metrics_file_name, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=SORTED_METRICS_FIELDS)
            if csvfile.tell() == 0:
                writer.writeheader()   
                
    def get_logger(self):
        return self.logger       
    
    def submit_mpi_task(self, script_path=None, num_procs=None, *args):
        return self.submit_task(run_mpi_task, num_procs, script_path, *args)
        
    def submit_task(self, func, *args, **kwargs):
        task_future = None
        try:
            pilot_scheduled = 'ANY'
            
            if "pilot" in kwargs:
                pilot_scheduled = kwargs["pilot"]
                del kwargs["pilot"]

            task_name = kwargs.get("task_name", f"task-{uuid.uuid4()}")
            if kwargs.get("task_name"):
                del kwargs["task_name"]

            if not self.client:
                self.client = self.get_client()

            if self.client is None:
                raise PilotAPIException("Cluster client isn't ready/provisioned yet")

            self.logger.info(f"Running task {task_name} on pilot {pilot_scheduled} with details func:{func.__name__}")
            
            task_metrics = copy(METRICS)
            task_metrics["task_id"] = task_name
            task_metrics["pilot_scheduled"] = pilot_scheduled
            task_metrics["submit_time"] = datetime.now()
            task_metrics["status"] = "RUNNING"
            
            
            def task_func(metrics_fn, *args, **kwargs):
                task_metrics["wait_time_secs"] = (datetime.now()-task_metrics["submit_time"]).total_seconds()
                
                task_execution_start_time = time.time()
                result = None
                
                try:
                    result = func(*args, **kwargs)
                    task_metrics["status"] = "SUCCESS"
                except Exception as e:
                    task_metrics["status"] = "FAILED"
                    task_metrics["error_msg"] = str(e)
                    

                task_metrics["completion_time"] = datetime.now()
                task_metrics["execution_secs"] = round((time.time() - task_execution_start_time), 4)

                lock = threading.Lock()
                
                with lock:
                    with open(metrics_fn, 'a', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=SORTED_METRICS_FIELDS)
                        writer.writerow(task_metrics)

                return result             
            

            if self.execution_engine == ExecutionEngine.DASK:
                if pilot_scheduled != 'ANY':
                    # find all the wokers in the pilot
                    workers = self.client.scheduler_info()['workers']
                    pilot_workers = [workers[worker]['name'] for worker in workers if workers[worker]['name'].startswith(pilot_scheduled)]                    
                    task_future = self.client.submit(task_func, self.metrics_file_name, *args, **kwargs, workers=pilot_workers)
                else:                
                    task_future = self.client.submit(task_func, self.metrics_file_name, *args, **kwargs)
                    time.sleep(1)
            elif self.execution_engine == ExecutionEngine.RAY:
                # Extract resource options from kwargs (if any)
                resources = kwargs.pop('resources', {})
                # staging_start_time = time.time()
                # args = [ray.put(arg) for arg in args]
                # kwargs = {key: ray.put(value) for key, value in kwargs.items()}
                # staging_end_time = time.time()
                # task_metrics["staging_time_secs"] = staging_end_time - staging_start_time
                # task_metrics["input_staging_data_size_bytes"] = sum([arg.size() for arg in args])
                task_future = ray.remote(task_func).options(**resources).remote(self.metrics_file_name, *args, **kwargs)                    
        except Exception as e:
            self.logger.error(f"Error submitting task {task_name} with details func:{func.__name__} - {str(e)}")
            raise PilotAPIException(f"Error submitting task {task_name} with details func:{func.__name__} - {str(e)}")
        
        return task_future
    


    def task(self, func):
        def wrapper(*args, **kwargs):
            return self.submit_task(func, *args, **kwargs)

        return wrapper

    def run(self, func, *args, **kwargs):
        if not self.client:
            self.client = self.get_client()

        if self.client is None:
            raise PilotAPIException("Cluster client isn't ready/provisioned yet")

        print(f"Running qtask with args {args}, kwargs {kwargs}")
        wrapper_func = self.task(func)
        return wrapper_func(*args, **kwargs).result()
    
    def wait_tasks(self, tasks):
        self.cluster_manager.wait_tasks(tasks)

    def get_results(self, tasks):
        return self.cluster_manager.get_results(tasks)    


class PilotComputeService(PilotComputeBase):
    def __init__(self, execution_engine, working_directory="/tmp"):
        self.execution_engine = execution_engine        
        self.pcs_working_directory = f"{working_directory}/pcs-{uuid.uuid4()}"                
        super().__init__(self.execution_engine, self.pcs_working_directory)
        self.logger.info(f"Initializing PilotComputeService with execution engine {execution_engine} and working directory {self.pcs_working_directory}")
        
        self.cluster_manager = self.__get_cluster_manager(execution_engine, self.pcs_working_directory)
        
        scheduler_start_time = time.time()
        self.cluster_manager.start_scheduler()
        scheduer_end_time = time.time()
        self.logger.info(f"[Metrics]scheduler_startup_metric_secs:{scheduer_end_time - scheduler_start_time}")
        self.logger.info("PilotComputeService initialized.")
        self.pilots = {}
        self.client = None


    def create_pilot(self, pilot_compute_description):
        pilot_submission_start_time = time.time()
        pilot_name = pilot_compute_description.get("name", f"pilot-{uuid.uuid4()}")
        pilot_compute_description["name"] = pilot_name

        self.logger.info(f"Create Pilot with description {pilot_compute_description}")
        pilot_compute_description["working_directory"] = self.pcs_working_directory

        batch_job = self.cluster_manager.submit_pilot(pilot_compute_description)
        self.pilot_id = batch_job.get_id()

        details = self.cluster_manager.get_config_data()
        self.logger.info(f"Cluster details: {details}")
        pilot = PilotCompute(batch_job, cluster_manager=self.cluster_manager)

        self.pilots[pilot_name] = pilot
        pilot_submission_end_time = time.time()
        self.logger.info(f"[Metrics]pilot_submission_metric_secs:{pilot_submission_end_time - pilot_submission_start_time}")
        return pilot

    def __get_cluster_manager(self, execution_engine, working_directory):
        if execution_engine == ExecutionEngine.DASK:
            # return dask_cluster_manager.Manager(working_directory)  # Replace with appropriate manager
            return dask_cluster_manager.DaskManager(working_directory)  # Replace with appropriate manager
        elif execution_engine == ExecutionEngine.RAY:
            # job_id = f"ray-{uuid.uuid1()}"
            return ray_cluster_manager.RayManager(working_directory)  # Replace with appropriate manager

        self.logger.error(f"Invalid Pilot Compute Description: invalid type: {execution_engine}")
        raise PilotAPIException(f"Invalid Pilot Compute Description: invalid type: {execution_engine}")

    def get_client(self):
        return self.cluster_manager.get_client()

    def get_pilots(self):
        return list(self.pilots.keys())
    
    def get_pilot(self, name):
        if name not in self.pilots:
            raise PilotAPIException(f"Pilot {name} not found")
        
        return self.pilots[name]

    def cancel(self):
        """Cancel the PilotComputeService.

        This also cancels all the PilotJobs under the control of this PJS.

        Returns:
        Result of the operation.
        """
        self.logger.info("Cancelling PilotComputeService.")
        self.cluster_manager.cancel()
        self.logger.info("Terminating scheduler ....")

        for pilot_name, pilot in self.pilots.items():
            self.logger.info(f"Terminating pilot {pilot_name} ....")
            pilot.cancel()    




class PilotCompute(PilotComputeBase):
    def __init__(self, batch_job=None, cluster_manager=None):
        super().__init__(cluster_manager.execution_engine, cluster_manager.working_directory)
        self.batch_job = batch_job
        self.cluster_manager = cluster_manager
        self.client = None

    def cancel(self):
        if self.cluster_manager:
            self.cluster_manager.cancel()
        if self.batch_job:
            self.batch_job.cancel()

    def get_state(self):
        if self.batch_job:
            return self.batch_job.get_state()

    def get_id(self):
        return self.cluster_manager.get_id()

    def get_details(self):
        return self.cluster_manager.get_config_data()

    def get_client(self):
        return self.cluster_manager.get_client()

    def wait(self):
        self.cluster_manager.wait()

    def submit_task(self, func, *args, **kwargs):
        kwargs["pilot"] = self.get_id()
        return super().submit_task(func, *args, **kwargs)


class PilotFuture:
    def __init__(self, future: Future):
        self._future = future

    def result(self):
        return self._future.result()

    def cancel(self):
        self._future.cancel()

    def done(self):
        return self._future.done()

    def exception(self):
        return self._future.exception()

    def add_done_callback(self, fn):
        self._future.add_done_callback(fn)

    def cancelled(self):
        return self._future.cancelled()

    def retry(self):
        self._future.retry()

    def release(self):
        self._future.release()

    def __repr__(self):
        return f"PilotFuture({self._future})"

    # Add additional methods or properties specific to your use case
    def custom_method(self):
        # Custom logic or functionality
        pass


