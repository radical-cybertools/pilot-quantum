import logging
import uuid

from distributed import Future

from pilot.plugins.dask import cluster as dask_cluster_manager
from pilot.plugins.ray import cluster as ray_cluster_manager

logging.basicConfig(level=logging.DEBUG)
from pennylane import numpy as np


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
        self.logger = logging.getLogger(__name__)
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
        resource_url = pilot_compute_description.get("resource", "")
        working_directory = pilot_compute_description.get("working_directory", "/tmp")
        project = pilot_compute_description.get("project")
        reservation = pilot_compute_description.get("reservation")
        queue = pilot_compute_description.get("queue")
        wall_time = pilot_compute_description.get("wall_time", 10)
        number_cores = int(pilot_compute_description.get("number_cores", 1))
        number_of_nodes = int(pilot_compute_description.get("number_of_nodes", 1))
        cores_per_node = int(pilot_compute_description.get("cores_per_node", 1))
        config_name = pilot_compute_description.get("config_name", "default")
        parent = pilot_compute_description.get("parent")

        framework_type = pilot_compute_description.get("type")
        if framework_type is None:
            self.logger.error("Invalid Pilot Compute Description: type not specified")
            raise PilotAPIException("Invalid Pilot Compute Description: type not specified")

        manager = self.__get_cluster_manager(framework_type, working_directory)

        batch_job = manager.submit_job(pilot_compute_description)

        details = manager.get_config_data()
        self.logger.info(f"Cluster details: {details}")
        pilot = PilotCompute(batch_job, cluster_manager=manager)
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

    def __init__(self, saga_job=None, cluster_manager=None):
        self.saga_job = saga_job
        self.cluster_manager = cluster_manager
        self.client = None

    def cancel(self):
        # self.cluster_manager.cancel()

        if self.saga_job:
            self.saga_job.cancel()

    def submit_task(self, func, *args, **kwargs):
        if not self.client:
            self.client = self.get_client()

        if self.client is None:
            raise PilotAPIException("Cluster client isn't ready/provisioned yet")

        print(f"Running qtask with args {args}, kwargs {kwargs}")
        return self.client.submit(func, *args, **kwargs)
        # return PilotFuture(future)

    def task(self, func):
        def wrapper(*args, **kwargs):
            return self.client.submit(func, *args, **kwargs)

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
        if self.saga_job:
            return self.saga_job.get_state()

    def get_id(self):
        return self.cluster_manager.get_jobid()

    def get_details(self):
        return self.cluster_manager.get_config_data()

    def get_client(self):
        return self.cluster_manager.get_client()

    def wait(self):
        self.cluster_manager.wait()

    def get_context(self, configuration=None):
        return self.cluster_manager.get_context(configuration)


class PilotFuture:
    def __init__(self, future: Future):
        self._future = future

    def result(self):
        result = self._future.result()
        type = self._future.type
        if type is np.tensor:
            return np.array(result, requires_grad=True)
        else:
            return result

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


def dask_submit(dask_pilot):
    def decorator(func):
        def wrapper(*args, **kwargs):
            return dask_pilot.submit_task(func, *args, **kwargs)

        return wrapper

    return decorator
