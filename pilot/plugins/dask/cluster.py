"""

Dask Cluster Manager
"""

import asyncio
import json
import logging
import os
import subprocess
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

from dask.distributed import Scheduler
import psutil
import time


logging.getLogger("tornado.application").setLevel(logging.CRITICAL)
logging.getLogger("distributed.utils").setLevel(logging.CRITICAL)


class Manager:
    def __init__(self, pcs_working_directory, job_id=None):
        self.pcs_working_directory = pcs_working_directory
        self.scheduler_info_file=f'{self.pcs_working_directory}/dask_scheduler'
        self.scheduler = None
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

    def stop_existing_processes(self, process_name):
        # Find the process IDs of all running dask-scheduler processes
        try:
            result = subprocess.run(['pgrep', '-f', process_name], stdout=subprocess.PIPE, text=True)
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    print(f"Stopping existing dask process with PID: {pid}")
                    subprocess.run(['kill', '-9', pid])
        except Exception as e:
            print(f"Error stopping existing schedulers: {e}")

    def start_scheduler(self):
        # Stop any existing Dask schedulers
        self.stop_existing_processes('dask')

        # Start a new Dask scheduler in the background
        log_file = 'dask_scheduler.log'
        with open(log_file, 'w') as f:
            process = subprocess.Popen(['dask', 'scheduler'], stdout=f, stderr=subprocess.STDOUT)

        
        # Wait and read the log file to get the scheduler address
        scheduler_address = None
        timeout = 10  # seconds
        start_time = time.time()
        while time.time() - start_time < timeout:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                if "Scheduler at:" in line:
                    scheduler_address = line.strip().split()[-1]
                    break
            if scheduler_address:
                break
            time.sleep(1)

        if scheduler_address is None:
            raise RuntimeError("Failed to start Dask scheduler")
        
        print(f"Scheduler started at {scheduler_address}")
        
        # Write scheduler address to file
        scheduler_info = {
            'address': scheduler_address
        }
        with open(self.scheduler_info_file, 'w') as f:
            json.dump(scheduler_info, f)
        self.logger.info(f"Scheduler details written to {self.scheduler_info_file}")
        
        return process
  
    
    def get_scheduler_info_file(self):
        return self.scheduler_info_file


    def _setup_job(self, pilot_compute_description):
        resource_url = pilot_compute_description["resource"]
        url_scheme = urlparse(resource_url).scheme

        if url_scheme.startswith("slurm"):
            js = pilot.job.slurm.Service(resource_url)
            executable = "python"
            arguments = ["-m", "pilot.plugins.dask.bootstrap_dask", "-t", self.dask_worker_type]
            if "cores_per_node" in pilot_compute_description:
                arguments.extend(["-p", str(self.pilot_compute_description["cores_per_node"])])

            arguments.extend(["-s", self.scheduler_info_file])
            arguments.extend(["-n", self.pilot_compute_description['name']])

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
            self.pilot_compute_description['working_directory'] = os.path.join(self.pcs_working_directory, self.job_id)
            self.working_directory = self.pilot_compute_description['working_directory']
            self.dask_worker_type = self.pilot_compute_description.get("type", "dask")

            try:
                os.makedirs(self.working_directory)
            except Exception:
                pass

            job_service, job_description = self._setup_job(pilot_compute_description)

            self.job = job_service.create_job(job_description)
            self.job.run()
            self.batch_job_id = self.job.get_id()
            self.logger.info(f"Job: {self.batch_job_id} State: {self.job.get_state()}")

            if not job_service.resource_url.startswith("slurm"):
                self.start_dask_workers()  # run dask on cloud platforms

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

    # start dask workers using dask-worker command and scheduler address
    def start_dask_workers(self):

        # get dash scheduler address
        with open(self.scheduler_info_file, 'r') as f:
            self.host = json.load(f)['address']

        # Get the address
        self.logger.info(f"Starting Dask workers with scheduler address: {self.host}")

        #Start dask workers on all nodes in background and write the worker address to a file
        command = f"dask worker --nthreads {self.pilot_compute_description.get('cores_per_node', 1)} --name {self.pilot_compute_description['name']} --memory-limit 3GB {self.host} &"
        self.logger.info(f"Starting worker with command: {command}")
        subprocess.Popen(command, shell=True)

        # wait for workers to start
        time.sleep(2)

        # get scheduler info
        client = Client(self.host)
        self.logger.info(client.scheduler_info())
        self.dask_client = client

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
        # Stop the pilot jobs if any
        if self.job:
            self.job.cancel()

        time.sleep(2)

        # Stop the scheduler
        self.stop_existing_processes('dask')


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
        master_file = os.path.join(self.pcs_working_directory, "dask_scheduler")
        # read the master json file and return the contents
        with open(master_file, 'r') as f:
            master_host = json.load(f)["address"]

        master_host = urlparse(master_host).hostname

        if master_host is None:
            raise Exception("Scheduler not found")
        
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
        return os.path.exists(os.path.join(self.pcs_working_directory, "dask_scheduler"))


if __name__ == "__main__":
    # Example usage:
    manager = Manager("my_job", "/path/to/working/directory")
    manager.submit_job()
    manager.wait()
    manager.cancel()
