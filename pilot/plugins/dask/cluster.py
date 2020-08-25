"""
Dask Cluster Manager

Supports launch via SLURM, EC2/SSH, SSH

"""

import os
import sys
import logging
import time
from datetime import datetime

import distributed
import subprocess

logging.getLogger("tornado.application").setLevel(logging.CRITICAL)
logging.getLogger("distributed.utils").setLevel(logging.CRITICAL)

# Resource Managers supported by Dask Pilot-Streaming Plugin
import pilot.job.slurm
import pilot.job.ec2
import pilot.job.ssh
import pilot.job.os

from urllib.parse import urlparse


class Manager():

    def __init__(self, jobid, working_directory):
        self.jobid = jobid
        print("{}{}".format(self.jobid, working_directory))
        self.working_directory = os.path.join(working_directory, jobid)
        self.pilot_compute_description = None
        self.myjob = None  # SAGA Job
        self.local_id = None  # Local Resource Manager ID (e.g. SLURM id)
        self.dask_process = None
        self.job_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.job_output = open("pilotstreaming_os_dask_agent_output_" + self.job_timestamp + ".log", "w")
        self.job_error = open("pilotstreaming_os_dask_agent_output__agent_error_" + self.job_timestamp + ".log", "w")
        try:
            os.makedirs(self.working_directory)
        except:
            pass

    # Dask 1.20
    def submit_job(self,
                   resource_url="fork://localhost",
                   number_of_nodes=1,
                   number_cores=1,
                   cores_per_node=1,
                   spmd_variation=None,
                   queue=None,
                   walltime=None,
                   project=None,
                   reservation=None,
                   config_name="default",
                   extend_job_id=None,
                   pilot_compute_description=None
                   ):
        try:
            # create a job service for SLURM LRMS or EC2 Cloud
            url_schema = urlparse(resource_url).scheme
            js = None
            if url_schema.startswith("slurm"):
                js = pilot.job.slurm.Service(resource_url)
            elif url_schema.startswith("ec2"):
                js = pilot.job.ec2.Service(resource_url)
            elif url_schema.startswith("os"):
                js = pilot.job.os.Service(resource_url)
            elif url_schema.startswith("ssh"):
                js = pilot.job.ssh.Service(resource_url)
            else:
                print("Unsupported URL Schema: %s " % resource_url)
                return

            self.pilot_compute_description = pilot_compute_description

            # environment, executable & arguments
            # NOT used yet - hardcoded in EC2 / SSH plugin
            executable = "python"
            arguments = ["-m", "pilot.plugins.dask.bootstrap_dask", " -p ", str(cores_per_node)]
            if "dask_cores" in pilot_compute_description:
                arguments = ["-m", "pilot.plugins.dask.bootstrap_dask", " -p ",
                             str(pilot_compute_description["dask_cores"])]

            if extend_job_id != None:
                arguments = ["-m", "pilot.plugins.dask.bootstrap_dask", "-j", extend_job_id]
            logging.debug("Run %s Args: %s" % (executable, str(arguments)))

            # executable = "/bin/hostname" # not required - just starting vms
            # arguments = "" # not required - just starting vms
            jd = {
                "executable": executable,
                "arguments": arguments,
                "working_directory": self.working_directory,
                "output": "dask_job_%s.stdout" % self.jobid,
                "error": "dask_job_%s.stderr" % self.jobid,
                "number_of_nodes": number_of_nodes,
                "cores_per_node": cores_per_node,
                "project": project,
                "reservation": reservation,
                "queue": queue,
                "walltime": walltime,
                "pilot_compute_description": pilot_compute_description
            }

            self.myjob = js.create_job(jd)
            self.myjob.run()
            self.local_id = self.myjob.get_id()
            print("**** Job: " + str(self.local_id) + " State: %s" % (self.myjob.get_state()))
            if not url_schema.startswith("slurm"):  # Dask is started in SLURM script. This is for ec2, openstack and ssh adaptors
                self.run_dask()

            return self.myjob
        except Exception as ex:
            print("An error occurred: %s" % (str(ex)))

    def run_dask(self):
        """TODO Move Dask specific stuff into Dask plugin"""
        ## Update Mini Apps
        # for i in nodes:
        #    self.wait_for_ssh(i)
        #    command = "ssh -o 'StrictHostKeyChecking=no' -i {} {}@{} pip install --upgrade git+ssh://git@github.com/radical-cybertools/streaming-miniapps.git".format(self.pilot_compute_description["ec2_ssh_keyfile"],
        #                                        self.pilot_compute_description["ec2_ssh_username"],
        #                                        i)
        #    print("Host: {} Command: {}".format(i, command))
        #    install_process = subprocess.Popen(command, shell=True, cwd=self.working_directory)
        #    install_process.wait()

        ## Run Dask
        # command = "dask-ssh --remote-dask-worker distributed.cli.dask_worker %s"%(self.host)
        self.host = self.myjob.get_node_list()[0]
        command = "ssh -o 'StrictHostKeyChecking=no' %s@%s dask-ssh %s" % (
        self.pilot_compute_description["os_ssh_username"], self.host, "localhost")
        if "cores_per_node" in self.pilot_compute_description:
            # command = "dask-ssh --nthreads %s --remote-dask-worker distributed.cli.dask_worker %s"%\
            command = "ssh -o 'StrictHostKeyChecking=no' -l %s %s dask-ssh --nthreads %s %s" % \
                      (self.pilot_compute_description["os_ssh_username"], self.host,
                       str(self.pilot_compute_description["cores_per_node"]), "localhost")
        print("Start Dask Cluster: {0}".format(command))
        # status = subprocess.call(command, shell=True)
        for i in range(3):
            self.dask_process = subprocess.Popen(command, shell=True,
                                                 cwd=self.working_directory,
                                                 stdout=self.job_output,
                                                 stderr=self.job_error,
                                                 close_fds=True)
            time.sleep(10)
            if self.dask_process.poll is not None:
                with open(os.path.join(self.working_directory, "dask_scheduler"), "w") as master_file:
                    master_file.write(self.host + ":8786")
                break

    def wait(self):
        while True:
            state = self.myjob.get_state()
            logging.debug("**** Job: " + str(self.local_id) + " State: %s" % (state))
            if state.lower() == "running":
                logging.debug("looking for Dask startup state at: %s" % self.working_directory)
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
        self.myjob.cancel()

    def submit_compute_unit(function_name):
        pass

    def get_context(self, configuration=None):
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

        master_host = master.split(":")[0]
        details = {
            "master_url": "tcp://%s:8786" % master_host,
            "web_ui_url": "http://%s:8787" % master_host,
        }
        return details

    def print_config_data(self):
        details = self.get_config_data()
        print("Dask Scheduler: %s" % details["master_url"])

    def is_scheduler_started(self):
        logging.debug("Results of scheduler startup file check: %s" % str(
            os.path.exists(os.path.join(self.working_directory, "dask_scheduler"))))
        return os.path.exists(os.path.join(self.working_directory, "dask_scheduler"))