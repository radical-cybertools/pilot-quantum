import csv
import logging
import time
import uuid

from distributed import Future

from pilot.pcs_logger import PilotComputeServiceLogger
from pilot.plugins.dask import cluster as dask_cluster_manager
from pilot.plugins.ray import cluster as ray_cluster_manager
import os
from dask.distributed import wait
from datetime import datetime
from enum import Enum
import csv
import os
import time
import uuid
from datetime import datetime
from enum import Enum

ExecutionEngine = Enum('ExecutionEngine', ["RAY", "DASK"])




class ExecutionEngine(Enum):
    DASK = "dask"
    RAY = "ray"

class PilotAPIException(Exception):
    pass

class PilotComputeBase:
    def __init__(self, metrics_file_name):
        self.metrics_file_name = metrics_file_name
        self.client = None
        self.logger = PilotComputeServiceLogger()

    def submit_task(self, func, *args, **kwargs):
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

        self.logger.info(f"Running task {task_name} with details func:{func.__name__};args {args};kwargs {kwargs}")


        metrics = {
            'task_id': task_name,
            'pilot_scheduled': pilot_scheduled,
            'submit_time': datetime.now(),
            'wait_time_secs': None, 
            'completion_time': None,            
            'execution_ms': None,
            'status': None,
            'error_msg': None,
        }

        def task_func(metrics_fn, *args, **kwargs):
            metrics["wait_time_secs"] = (datetime.now()-metrics["submit_time"]).total_seconds()
            
            task_execution_start_time = time.time()
            try:
                result = func(*args, **kwargs)
                metrics["status"] = "SUCCESS"
            except Exception as e:
                metrics["status"] = "FAILED"
                metrics["error_msg"] = str(e)
                

            metrics["completion_time"] = datetime.now()
            metrics["execution_ms"] = time.time() - task_execution_start_time

            with open(metrics_fn, 'a', newline='') as csvfile:
                fieldnames = ['task_id','pilot_scheduled','submit_time', 'wait_time_secs', 'completion_time', 'execution_ms', 'status', 'error_msg']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(metrics)

            return result             
        

        if pilot_scheduled != 'ANY':
            # find all the wokers in the pilot
            workers = self.client.scheduler_info()['workers']
            pilot_workers = [workers[worker]['name'] for worker in workers if workers[worker]['name'].startswith(pilot_scheduled)]

            task_future = self.client.submit(task_func, self.metrics_file_name, *args, **kwargs, workers=pilot_workers)
        else:
            task_future = self.client.submit(task_func, self.metrics_file_name, *args, **kwargs)

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
        wait(tasks)

        for task in tasks:
            if task.done() and task.exception() is not None:                                
                self.logger.info(f"Task {task} completed {task.status} with exception: {task.exception()}")


class PilotComputeService(PilotComputeBase):
    def __init__(self, execution_engine=ExecutionEngine.DASK, working_directory="/tmp"):                
        self.pcs_working_directory = f"{working_directory}/pcs-{uuid.uuid4()}"
        os.makedirs(self.pcs_working_directory)

        super().__init__(os.path.join(self.pcs_working_directory, "metrics.csv"))
        self.execution_engine = execution_engine
        
        self.cluster_manager = self.__get_cluster_manager(execution_engine, self.pcs_working_directory)
        self.cluster_manager.start_scheduler()

        
        self.logger.info("PilotComputeService initialized.")
        self.pilots = {}
        self.client = None

        with open(self.metrics_file_name, 'a', newline='') as csvfile:
            fieldnames = ['task_id', 'pilot_scheduled', 'submit_time', 'wait_time_secs', 'completion_time',
                          'execution_ms', 'status', 'error_msg']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if csvfile.tell() == 0:
                writer.writeheader()

    def create_pilot(self, pilot_compute_description):
        pilot_name = pilot_compute_description.get("name", f"pilot-{uuid.uuid4()}")
        pilot_compute_description["name"] = pilot_name

        worker_cluster_manager = self.__get_cluster_manager(self.execution_engine, self.pcs_working_directory)

        self.logger.info(f"Create Pilot with description {pilot_compute_description}")
        pilot_compute_description["working_directory"] = self.pcs_working_directory

        batch_job = worker_cluster_manager.submit_job(pilot_compute_description)
        self.pilot_id = batch_job.get_id()

        details = worker_cluster_manager.get_config_data()
        self.logger.info(f"Cluster details: {details}")
        pilot = PilotCompute(self.metrics_file_name, batch_job, cluster_manager=worker_cluster_manager)

        self.pilots[pilot_name] = pilot
        return pilot

    def __get_cluster_manager(self, execution_engine, working_directory):
        if execution_engine == ExecutionEngine.DASK:
            return dask_cluster_manager.Manager(working_directory)  # Replace with appropriate manager
        elif execution_engine == ExecutionEngine.RAY:
            job_id = f"ray-{uuid.uuid1()}"
            return ray_cluster_manager.Manager(job_id, working_directory)  # Replace with appropriate manager

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
    def __init__(self, metrics_file_name, batch_job=None, cluster_manager=None):
        super().__init__(metrics_file_name)
        self.batch_job = batch_job
        self.cluster_manager = cluster_manager
        self.client = None

    def cancel(self):
        if self.client:
            self.client.close()
        if self.batch_job:
            self.batch_job.cancel()

    def get_state(self):
        if self.batch_job:
            return self.batch_job.get_state()

    def get_id(self):
        return self.cluster_manager.get_jobid()

    def get_details(self):
        return self.cluster_manager.get_config_data()

    def get_client(self):
        return self.cluster_manager.get_client()

    def wait(self):
        self.cluster_manager.wait()

    def submit_task(self, func, *args, **kwargs):
        kwargs["pilot"] = self.get_id()
        return super().submit_task(func, *args, **kwargs)


class PilotAPIException(Exception):
    pass

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


