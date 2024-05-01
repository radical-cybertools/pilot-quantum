"""
Ray Cluster Manager

Supports launch via SSH and SLURM.

"""
import getpass
import logging
import os
import sys
import time
from datetime import datetime
import uuid
import ray

import numpy as np

logging.getLogger("tornado.application").setLevel(logging.CRITICAL)
logging.getLogger("distributed.utils").setLevel(logging.CRITICAL)

# Resource Managers supported by Dask Pilot-quantum Plugin
import pilot.job.slurm
import pilot.job.ssh
from urllib.parse import urlparse

from pilot.util.ssh_utils import execute_ssh_command, execute_ssh_command_as_daemon

class Manager():

    def __init__(self, 
                 job_id, 
                 working_directory):
        
        self.job_id = job_id
        if not self.job_id:
            self.job_id = f"ray-{uuid.uuid1()}"
        #print("{}{}".format(self.job_id, working_directory))
        self.working_directory = os.path.join(working_directory, self.job_id)
        # create working directory if not exists
        if not os.path.exists(self.working_directory):
            os.makedirs(self.working_directory)

        self.start_agent_working_directory = working_directory
        self.pilot_compute_description = None
        self.job = None  # SAGA Job
        self.local_id = None  # Local Resource Manager ID (e.g. SLURM id)
        self.ray_process = None
        self.ray_cluster = None
        self.job_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.job_output = open(os.path.join(self.working_directory, 
                                            self.job_timestamp + "_ray_pilot_agent_output.log"), "w")
        self.job_error = open(os.path.join(self.working_directory, 
                                           self.job_timestamp + "_ray_pilot_agent_error.log"), "w")

        self.ray_client = None # Ray Client
        try:
            os.makedirs(self.working_directory)
        except:
            pass


    def _configure_logging(self):
        logging.basicConfig(
            filename=os.path.join(self.working_directory, "pilot_quantum_ray.log"),
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
    
    def _setup_job(self, pilot_compute_description):
        resource_url = pilot_compute_description["resource"]

        url_scheme = urlparse(resource_url).scheme

        if url_scheme.startswith("slurm"):
            js = pilot.job.slurm.Service(resource_url)
            executable = "python"
            arguments = ["-m", "pilot.plugins.ray.bootstrap_ray",
                         "-j", self.job_id]
            if "cores_per_node" in pilot_compute_description:
                arguments.append(" -p ")
                arguments.append(str(pilot_compute_description["cores_per_node"]))
            
            if  "gpus_per_node" in pilot_compute_description:            
                arguments.append(" -g ")
                arguments.append(str(pilot_compute_description["gpus_per_node"]))
            
            if "working_directory" in pilot_compute_description:
                arguments.append(" -w ")
                arguments.append(str(pilot_compute_description["working_directory"]))
            
            logging.debug("Run %s Args: %s" % (executable, str(arguments)))
        else:
            js = pilot.job.ssh.Service(resource_url)
            executable = "/bin/hostname"
            arguments = ""

        jd = {"executable": executable, "arguments": arguments}
        jd.update(pilot_compute_description)

        return js, jd

    # Ray 2.12.0
    def submit_job(self,
                   pilot_compute_description=None
                   ):
        # Start Pilot-Job via Job abstraction
        try:
            self.pilot_compute_description = pilot_compute_description
            self.pilot_compute_description["working_directory"] = os.path.join(self.pilot_compute_description["working_directory"], self.job_id)
            self.working_directory = self.pilot_compute_description["working_directory"]
            try:
                os.makedirs(self.working_directory)
            except:
                pass

            self._configure_logging()

            job_service, job_description = self._setup_job(pilot_compute_description)

            self.job = job_service.create_job(job_description)
            self.job.run()
            self.batch_job_id = self.job.get_id()
            logging.info("Job: %s State: %s", str(self.batch_job_id), self.job.get_state())
          
            return self.job

        except Exception as ex:
            print("An error occurred: %s" % (str(ex)))
            raise ex
   

    def wait(self):
        while True:
            state = self.job.get_state()
            logging.debug("**** Job: " + str(self.local_id) + " State: %s" % (state))
            if state.lower() == "running":
                logging.debug("Looking for Ray startup state at: %s" % self.working_directory)
                if self.is_scheduler_started():
                    for i in range(5):
                        try:
                            print("init Ray client")
                            c = self.get_context()
                            #print(str(c.address_info))
                            return c
                        except IOError as e:
                            print("Ray Client Connect Attempt {} failed".format(i))
                            time.sleep(5)
            elif state == "Failed":
                break
            time.sleep(6)

    def cancel(self):
        # c = self.get_context()
        # ray.client().disconnect()
        # ray.shutdown()
        self.job.cancel()

    def submit_compute_unit(function_name):
        pass

    def get_client(self):
        return self.get_context()

    def get_context(self, configuration=None) -> object:
        """Returns Ray Client for Cluster"""
        details = self.get_config_data()
        if details is not None and ray.is_initialized() == False:
            print("Connect to %s"%details["master_url"])
            #self.ray_client = ray.util.connect(details["master_url"], 
            #                                   allow_multiple=True)
            self.ray_client = ray.init(
                address="ray://%s"%details["master_url"], 
                allow_multiple=True)
            # try:
            #     print("Connect to Ray DIRECT: %s" % details["master_url"])            
            #     # create a ray client for cluster url            
            #     self.ray_client  = ray.init(address=details["master_url"],
            #                   ignore_reinit_error=True,
            #                   timeout=10) # RayContext
            #     return self.ray_client 
            # except Exception as ex:
            #     client_address = details["master_url"].replace("6379", "10001")
            #     print("Connect to Ray Client: %s" % client_address)            
            #     # create a ray client for cluster url            
            #     self.ray_client  = ray.init(address="ray://" + 
            #                       client_address,
            #                   ignore_reinit_error=True,
            #                   allow_multiple=True)  # ClientContext
            #    return self.ray_client 
        return self.ray_client 

    def get_jobid(self):
        return self.job_id

    def get_config_data(self):
        if not self.is_scheduler_started():
            logging.debug("Scheduler not started")
            return None
        master_file = os.path.join(self.working_directory, "ray_scheduler")
        # print master_file
        master = "localhost"
        counter = 0
        while os.path.exists(master_file) == False and counter < 600:
            time.sleep(2)
            counter = counter + 1

        with open(master_file, 'r') as f:
            master = f.read()

        if master.startswith("ray://"):
            details = {
                "master_url": master
            }
        else:
            master_host = master.split(":")[0]
            details = {
                "master_url": "%s:10001" % master_host,
                "web_ui_url": "http://%s:8265" % master_host,
            }
        return details

    def print_config_data(self):
        details = self.get_config_data()
        print("Ray Scheduler: %s" % details["master_url"])

    def is_scheduler_started(self):
        logging.debug("Results of scheduler startup file check: %s" % str(
            os.path.exists(os.path.join(self.working_directory, "ray_scheduler"))))
        return os.path.exists(os.path.join(self.working_directory, "ray_scheduler"))
