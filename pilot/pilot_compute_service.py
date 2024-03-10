import logging
import uuid

from pilot.plugins.dask import cluster as dask_cluster_manager
from pilot.plugins.ray import cluster as ray_cluster_manager

logging.basicConfig(level=logging.DEBUG)


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

        if framework_type.startswith("dask"):
            batch_job = manager.submit_job(pilot_compute_description)
        else:
            batch_job = manager.submit_job(
                resource_url=resource_url,
                number_of_nodes=number_of_nodes,
                number_cores=number_cores,
                cores_per_node=cores_per_node,
                queue=queue,
                walltime=wall_time,
                project=project,
                reservation=reservation,
                config_name=config_name,
                extend_job_id=parent,
                pilot_compute_description=pilot_compute_description
            )


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

    def cancel(self):
        # self.cluster_manager.cancel()

        if self.saga_job:
            self.saga_job.cancel()

    def submit(self, function_name):
        self.cluster_manager.submit_compute_unit(function_name)

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
