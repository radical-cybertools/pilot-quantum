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


class PilotComputeService:
    """PilotComputeService (PCS) for creating and managing PilotComputes.

    The PilotComputeService is the application's interface to the Pilot-Manager
    in the P* Model.
    """

    def __init__(self):
        """Create a PilotComputeService object.

        Args:
        pjs_id (optional): Connect to an existing PilotComputeService.
        """
        self.logger = PilotComputeServiceLogger()        
        self.logger.info("PilotComputeService initialized.")

    def cancel(self):
        """Cancel the PilotComputeService.

        This also cancels all the PilotJobs under the control of this PJS.

        Returns:
        Result of the operation.
        """
        self.logger.info("Cancelling PilotComputeService.")
        pass

    def create_pilot(self, pilot_compute_description):
        """
        Create and initialize a PilotCompute instance based on the provided description.

        :param pilot_compute_description: Dictionary containing details about the cluster to launch.
        :return: Initialized PilotCompute instance.
        """
        self.logger.info("Creating a new PilotCompute.")
        working_directory = pilot_compute_description.get("working_directory", "/tmp")
        
        framework_type = pilot_compute_description.get("type")
        if framework_type is None:
            self.logger.error("Invalid Pilot Compute Description: type not specified")
            raise PilotAPIException("Invalid Pilot Compute Description: type not specified")

        manager = self.__get_cluster_manager(framework_type, working_directory)

        batch_job = manager.submit_job(pilot_compute_description)
        self.pilot_id = batch_job.get_id()

        self.metrics_file_name = os.path.join(working_directory, f"{self.pilot_id}-metrics.csv")


        details = manager.get_config_data()
        self.logger.info(f"Cluster details: {details}")
        pilot = PilotCompute(self.metrics_file_name, batch_job, cluster_manager=manager)
        return pilot
    


    def __get_cluster_manager(self, framework_type, working_directory):
        """
        Get the appropriate ClusterManager based on the framework type.

        :param framework_type: Type of the computing framework.
        :param working_directory: Working directory for the cluster.
        :return: ClusterManager instance.
        """
        if framework_type.startswith("dask"):
            return dask_cluster_manager.Manager()  # Replace with appropriate manager
        elif framework_type == "ray":
            job_id = f"ray-{uuid.uuid1()}"
            return ray_cluster_manager.Manager(job_id, working_directory)  # Replace with appropriate manager

        self.logger.error(f"Invalid Pilot Compute Description: invalid type: {framework_type}")
        raise PilotAPIException(f"Invalid Pilot Compute Description: invalid type: {framework_type}")

    def task(self, func):
        """
        Submit task to PilotComputeService, which can be scheduled on a group of pilots.
        """
        def wrapper(*args, **kwargs):
            pass

        return wrapper




class PilotAPIException(Exception):
    pass


class PilotCompute(object):
    """PilotCompute (PC) representation.

    This class is returned by the PilotComputeService when a new PilotCompute
    (aka Pilot-Job) is created based on a PilotComputeDescription.

    The PilotCompute object can be used by the application to keep track
    of active PilotComputes. It has state, can be queried, can be cancelled,
    and be re-initialized.
    """

    def __init__(self, metrics_file_name, batch_job=None, cluster_manager=None):
        self.batch_job = batch_job
        self.cluster_manager = cluster_manager
        self.client = None
        self.metrics_fn = metrics_file_name


    def cancel(self):
        if self.client:
            self.client.close()
        if self.batch_job:
            self.batch_job.cancel()

    def submit_task(self, task_name, func, *args, **kwargs):
        if not self.client:
            self.client = self.get_client()

        if self.client is None:
            raise PilotAPIException("Cluster client isn't ready/provisioned yet")

        print(f"Running task {task_name} with details func:{func.__name__};args {args};kwargs {kwargs}")


        metrics = {
            'task_id': task_name,
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
                fieldnames = ['task_id', 'submit_time', 'wait_time_secs', 'completion_time', 'execution_ms', 'status', 'error_msg']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                if csvfile.tell() == 0:
                    writer.writeheader()

                writer.writerow(metrics)

            return result             

        task_future = self.client.submit(task_func, self.metrics_fn, *args, **kwargs)

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

    def get_state(self):
        """
        Get the state of the PilotCompute.
        """
        if self.batch_job:
            return self.batch_job.get_state()

    def get_id(self):
        return self.cluster_manager.get_jobid()

    def get_details(self):
        return self.cluster_manager.get_config_data()

    def get_client(self):
        """
        Returns the native client for interacting with the task execution engine (i.e. Dask or Ray) started via the Pilot-Job.
        see also get_context()
        """
        return self.cluster_manager.get_client()

    def wait(self):
        self.cluster_manager.wait()

    def wait_tasks(self, tasks):
        wait(tasks)
        

    def get_context(self, configuration=None):
        """
        Returns the context for interacting with the task execution engine (i.e. Dask or Ray) started via the Pilot-Job.
        """
        return self.cluster_manager.get_context(configuration)


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


