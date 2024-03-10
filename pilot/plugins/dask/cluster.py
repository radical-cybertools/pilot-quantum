"""
Dask Cluster Manager
"""

import getpass
import logging
import os
import sys
import time
from datetime import datetime
from urllib.parse import urlparse
import uuid

import distributed
import numpy as np
from dask.distributed import Client, SSHCluster

# Resource Managers supported by Dask Pilot-quantum Plugin
import pilot.job.slurm
import pilot.job.ssh

logging.getLogger("tornado.application").setLevel(logging.CRITICAL)
logging.getLogger("distributed.utils").setLevel(logging.CRITICAL)


class Manager:
    def __init__(self, job_id=None):
        self.dask_client = None
        self.job_id = job_id
        if not self.job_id:
            self.job_id = f"dask-{uuid.uuid1()}"



    def _configure_logging(self):
        logging.basicConfig(
            filename=os.path.join(self.working_directory, "pilot_quantum_dask.log"),
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def _setup_job(self, pilot_compute_description):
        resource_url = pilot_compute_description["resource"]

        url_scheme = urlparse(resource_url).scheme

        if url_scheme.startswith("slurm"):
            js = pilot.job.slurm.Service(resource_url)
            executable = "python"
            arguments = ["-m", "pilot.plugins.dask.bootstrap_dask", "-t", self.dask_worker_type]
            if "dask_cores" in pilot_compute_description:
                arguments = ["-m", "pilot.plugins.dask.bootstrap_dask", "-p", "-t", self.dask_worker_type]

            logging.debug("Run %s Args: %s", executable, str(arguments))
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
            self.pilot_compute_description["working_directory"] = os.path.join(self.pilot_compute_description["working_directory"], self.job_id)
            self.working_directory = self.pilot_compute_description["working_directory"]
            self.dask_worker_type = self.pilot_compute_description["type"]

            os.makedirs(self.working_directory)
            self._configure_logging()

            job_service, job_description = self._setup_job(pilot_compute_description)

            self.job = job_service.create_job(job_description)
            self.job.run()
            self.batch_job_id = self.job.get_id()
            logging.info("Job: %s State: %s", str(self.batch_job_id), self.job.get_state())



            if not job_service.resource_url.startswith("slurm"):
                self.run_dask()  # run dask on cloud platforms

            return self.job
        except Exception as ex:
            logging.error("An error occurred: %s", str(ex))
            raise ex

    def run_dask(self):
        ## Run Dask
        # command = "dask-ssh --remote-dask-worker distributed.cli.dask_worker %s"%(self.host)
        self.nodes = self.job.get_nodes_list()
        resource_url = self.pilot_compute_description["resource"]
        # self.host = self.myjob.get_nodes_list_public()[0] #first node is master host - requires public ip to connect to
        self.host = self.nodes[0]  # first node is master host - requires public ip to connect to
        self.user = None
        print("Check for user name")
        if urlparse(resource_url).username is not None:
            self.user = urlparse(resource_url).username
            self.pilot_compute_description["os_ssh_username"] = self.user

        elif "os_ssh_username" in self.pilot_compute_description:
            self.user = self.pilot_compute_description["os_ssh_username"]
        else:
            self.user = getpass.getuser()
            self.pilot_compute_description["os_ssh_username"] = self.user

        print("Check for user name*****", self.user)

        print("Check for ssh key")
        self.ssh_key = "~/.ssh/mykey"
        try:
            if "os_ssh_keyfile" in self.pilot_compute_description["os_ssh_keyfile"]:
                self.ssh_key = self.pilot_compute_description["os_ssh_keyfile"]
        except:
            # set to default key for further processing
            self.pilot_compute_description["os_ssh_keyfile"] = self.ssh_key

        worker_options = {"nthreads": 1, "n_workers": 1, "memory_limit": '3GB'}
        try:
            if "cores_per_node" in self.pilot_compute_description:
                # dask_command = 'dask-ssh --ssh-private-key {} --nthreads {} {}'.format(self.ssh_key, self.user, self.pilot_compute_description["cores_per_node"], " ".join(self.nodes))
                worker_options = {"nthreads": 1,
                                  "n_workers": self.pilot_compute_description["cores_per_node"],
                                  "memory_limit": '3GB'}
        except:
            pass

        hosts = list(np.append(self.nodes[0], self.nodes))
        print("Connecting to hosts", hosts)
        self.dask_cluster = SSHCluster(hosts,
                                       connect_options={"known_hosts": None, "username": self.user},
                                       worker_options=worker_options,
                                       scheduler_options={"port": 0, "dashboard_address": ":8797"})
        client = Client(self.dask_cluster)
        print(client.scheduler_info())
        self.host = client.scheduler_info()["address"]

        if self.host is not None:
            with open(os.path.join(self.working_directory, "dask_scheduler"), "w") as master_file:
                master_file.write(self.host)

    def wait(self):
        while True:
            state = self.job.get_state()
            logging.debug("**** Job: " + str(self.batch_job_id) + " State: %s" % (state))
            if state.lower() == "running":
                logging.debug("Looking for Dask startup state at: %s" % self.working_directory)
                if self.is_scheduler_started():
                    for i in range(5):
                        try:
                            print("init distributed client")
                            c = self.get_context()
                            # c.scheduler_info()
                            print(str(c.scheduler_info()))
                            c.close()

                            return
                        except IOError as e:
                            print("Dask Client Connect Attempt {} failed".format(i))
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
            print("Connect to Dask: %s" % details["master_url"])
            client = distributed.Client(details["master_url"])
            return client
        return None

    def get_jobid(self):
        return self.jobid

    def get_config_data(self):
        if not self.is_scheduler_started():
            logging.debug("Scheduler not started")
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
        logging.info("Dask Scheduler: %s", details["master_url"])

    def is_scheduler_started(self):
        return os.path.exists(os.path.join(self.working_directory, "dask_scheduler"))


if __name__ == "__main__":
    # Example usage:
    manager = Manager("my_job", "/path/to/working/directory")
    manager.submit_job()
    manager.wait()
    manager.cancel()
