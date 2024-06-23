"""

Dask Cluster Manager
"""

import logging
import os
import sys
import time
import uuid
from urllib.parse import urlparse

import distributed
import numpy as np
from dask.distributed import Client, SSHCluster

# Resource Managers supported by Dask Pilot-quantum Plugin
import pilot.job.slurm
import pilot.job.ssh
from pilot.pcs_logger import PilotComputeServiceLogger

logging.getLogger("tornado.application").setLevel(logging.CRITICAL)
logging.getLogger("distributed.utils").setLevel(logging.CRITICAL)


class Manager:
    def __init__(self, job_id=None):
        self.host = None
        self.dask_cluster = None
        self.batch_job_id = None
        self.job = None
        self.logger = PilotComputeServiceLogger()
        self.dask_worker_type = None
        self.working_directory = None
        self.pilot_compute_description = None
        self.nodes = None
        self.dask_client = None
        self.job_id = job_id
        if not self.job_id:
            self.job_id = f"dask-{uuid.uuid1()}"

    def _setup_job(self, pilot_compute_description):
        resource_url = pilot_compute_description["resource"]
        url_scheme = urlparse(resource_url).scheme

        if url_scheme.startswith("slurm"):
            js = pilot.job.slurm.Service(resource_url)
            executable = "python"
            arguments = ["-m", "pilot.plugins.dask.bootstrap_dask", "-t", self.dask_worker_type]
            if "cores_per_node" in pilot_compute_description:
                arguments.extend(["-p", str(self.pilot_compute_description["cores_per_node"])])

            self.logger.debug(f"Run {executable} Args: {arguments}")
        else:
            js = pilot.job.ssh.Service(resource_url)
            executable = "/bin/hostname"
            arguments = ""

        jd = {"executable": executable, "arguments": arguments}
        jd.update(pilot_compute_description)

        return js, jd


    def submit_job(self, pilot_compute_description):
        try:
            self.pilot_compute_description = pilot_compute_description
            self.pilot_compute_description["working_directory"] = os.path.join(
                self.pilot_compute_description["working_directory"], self.job_id)
            self.working_directory = self.pilot_compute_description["working_directory"]
            self.dask_worker_type = self.pilot_compute_description["type"]

            os.makedirs(self.working_directory)

            job_service, job_description = self._setup_job(pilot_compute_description)

            self.job = job_service.create_job(job_description)
            self.job.run()
            self.batch_job_id = self.job.get_id()
            self.logger.info(f"Job: {self.batch_job_id} State: {self.job.get_state()}")

            if not job_service.resource_url.startswith("slurm"):
                self.run_dask()  # run dask on cloud platforms

            return self.job
        except Exception as ex:
            self.logger.error(f"An error occurred: {str(ex)}")
            raise ex

    def run_dask(self):
        self.nodes = self.job.get_nodes_list()
        self.host = self.nodes[0]  # first node is master host - requires public ip to connect to

        worker_options = {"nthreads": self.pilot_compute_description.get("cores_per_node", 1), 
                          "n_workers": self.pilot_compute_description.get("number_of_nodes", 1),
                          "memory_limit": '3GB'}
        hosts = list(np.append(self.nodes[0], self.nodes))
        self.dask_cluster = SSHCluster(hosts,
                                       connect_options={"known_hosts": None},
                                       worker_options=worker_options,
                                       scheduler_options={"port": 0, "dashboard_address": ":8797"})
        client = Client(self.dask_cluster)
        self.logger.info(client.scheduler_info())
        self.host = client.scheduler_info()["address"]

        if self.host is not None:
            with open(os.path.join(self.working_directory, "dask_scheduler"), "w") as master_file:
                master_file.write(self.host)

    def wait(self):
        while True:
            state = self.job.get_state()
            self.logger.debug("**** Job: " + str(self.batch_job_id) + " State: %s" % (state))
            if state.lower() == "running":
                self.logger.debug("Looking for Dask startup state at: %s" % self.working_directory)
                if self.is_scheduler_started():
                    for i in range(5):
                        try:
                            self.logger.info("init distributed client")
                            c = self.get_context()
                            scheduler_info = c.scheduler_info()
                            if len(scheduler_info.get('workers')) > 0:
                                self.logger.info(str(c.scheduler_info()))
                                c.close()
                                return
                            else:
                                self.logger.info("Dask cluster is still initializing. Waiting...")
                                time.sleep(5)
                        except IOError as e:
                            self.logger.warning("Dask Client Connect Attempt {} failed".format(i))
                            time.sleep(5)
            elif state == "Failed":
                break
            time.sleep(6)

    def cancel(self):
        c = self.get_context()
        c.run_on_scheduler(lambda dask_scheduler=None: dask_scheduler.close() & sys.exit(0))
        self.dask_cluster.close()
        # self.myjob.cancel()

    def submit_compute_unit(function_name):
        pass

    def get_context(self, configuration=None) -> object:
        """Returns Dask Client for Scheduler"""
        details = self.get_config_data()
        if details is not None:
            self.logger.info("Connect to Dask: %s" % details["master_url"])
            client = distributed.Client(details["master_url"])
            return client
        return None

    def get_jobid(self):
        return self.jobid

    def get_config_data(self):
        if not self.is_scheduler_started():
            self.logger.debug("Scheduler not started")
            return None
        master_file = os.path.join(self.working_directory, "dask_scheduler")
        # print master_file
        master = "localhost"
        counter = 0
        while os.path.exists(master_file) == False and counter < 600:
            time.sleep(2)
            counter = counter + 1

        with open(master_file, 'r') as f:
            master = f.read()

        if master.startswith("tcp://"):
            details = {
                "master_url": master
            }
        else:
            master_host = master.split(":")[0]
            details = {
                "master_url": "tcp://%s:8786" % master_host,
                "web_ui_url": "http://%s:8787" % master_host,
            }
        return details

    def get_client(self):
        if self.dask_client:
            return self.dask_client
        return distributed.Client(self.get_config_data()['master_url'])

    def print_config_data(self):
        details = self.get_config_data()
        self.logger.info("Dask Scheduler: %s", details["master_url"])

    def is_scheduler_started(self):
        return os.path.exists(os.path.join(self.working_directory, "dask_scheduler"))


if __name__ == "__main__":
    # Example usage:
    manager = Manager("my_job", "/path/to/working/directory")
    manager.submit_job()
    manager.wait()
    manager.cancel()
